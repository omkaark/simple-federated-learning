import io
import os
import socket
import threading
import grpc
from concurrent import futures
import time
import argparse
import torch
import logging
import gc
from protos import learner_pb2, learner_pb2_grpc
from protos import leader_pb2, leader_pb2_grpc
from utils import get_data_loader
from model_artifacts.model import model, optimizer_function, device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GRPC_STUB_OPTIONS = [
    ('grpc.max_send_message_length', 50 * 1024 * 1024),  # For example, 50 MB
    ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # For example, 50 MB
]

class LeaderService(leader_pb2_grpc.LeaderServiceServicer):
    def __init__(self, learner_count):
        self.lock = threading.Lock()
        self.max_learners = learner_count
        self.learner_list = []
        self.gradient_list = []
        self.global_model = model
        self.global_optimizer = optimizer_function(self.global_model)
        self.device = device
        self.num_batches = None
        self.accumulation_count = 0
        logging.info("Leader service initialized")

    def RegisterLearner(self, request, context):
        # lock needed to handle learner_list with thread safety
        with self.lock:
            logging.info(f"Registering learner: network_addr={request.network_addr}")
            if len(self.learner_list) < self.max_learners:
                # learners < expected, then register them
                new_id = len(self.learner_list)
                learner_stub = learner_pb2_grpc.LearnerServiceStub(
                    grpc.insecure_channel(request.network_addr, options=GRPC_STUB_OPTIONS)
                )
                # give it the data loader generator
                data_loader, num_batches = get_data_loader(learner_id=new_id, max_learners=self.max_learners)
                # knowing num_batches allows to know when training is done, all learners should have equal # od batches
                self.num_batches = num_batches
                self.learner_list.append(
                    {
                        'id': new_id,
                        'network_addr': request.network_addr,
                        'batches_consumed': 0, 
                        'stub': learner_stub,
                        'data_loader': data_loader,
                    }
                )
                # if learners = expected, then start training on all of them
                if len(self.learner_list) == self.max_learners:
                    thread = threading.Thread(target=self.start_training)
                    thread.start()
                return leader_pb2.AckWithMetadata(success=True, message="Registered learner", learner_id=new_id, max_learners=self.max_learners)
            else:
                return leader_pb2.Ack(success=False, message="Max learners reached")

    def start_training(self):
        time.sleep(3) # Start up time for the last learner
        logging.info("Starting training across all registered learners.")
        for learner in self.learner_list:
            learner['stub'].StartTraining(learner_pb2.Empty())

    def GetModel(self, request, context):
        logging.info("Sending model state to learner")
        model_dir = './model_artifacts'
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'model.py')
        
        # Stream the model file
        with open(model_path, 'rb') as model_file:
            while True:
                chunk = model_file.read(1024 * 1024)
                if not chunk:
                    break
                yield leader_pb2.ModelChunk(chunk=chunk)

    def GetData(self, request, context):
        logging.info(f"Sending data to learner network_addr {request.network_addr}")
        # get learner that requested the data
        learner = next((l for l in self.learner_list if l['network_addr'] == request.network_addr), None)
        if not learner:
            context.abort(grpc.StatusCode.NOT_FOUND, 'Learner not found')
            return
        
        # calculate the number of batches to send and the ending batch index
        start_index = learner['batches_consumed']

        # usually 10 batches are sent at a time to reduce network costs
        end_index = min(start_index + 10, self.num_batches)
        
        # check if there are batches left to send
        if start_index < self.num_batches:
            for i, (input, labels) in enumerate(learner['data_loader']):
                # send only data batches that fall in the start to end index range
                if start_index <= i < end_index:
                    buffer = io.BytesIO()
                    torch.save((input, labels), buffer)
                    buffer.seek(0)
                    serialized_batch = buffer.read()
                    yield leader_pb2.DataChunk(chunk=serialized_batch)
                    learner['batches_consumed'] += 1

            logging.info(f"Learner {learner['id']} is getting batches {start_index + 1} to {end_index} of {self.num_batches}")
        else:
            # when nothing is sent back to learner, the learner gracefully quits itself
            logging.info(f"No more batches to send to learner {learner['id']}.")

    def AccumulateGradients(self, request, context):
        # needs to be thread safe for gradient_list to handle accumulations properly
        with self.lock:
            logging.info("Accumulating gradients from learner")
            buffer = io.BytesIO(request.chunk)
            gradients = torch.load(buffer)
            self.gradient_list.append(gradients)
            
            if len(self.gradient_list) == len(self.learner_list):
                self.update_and_broadcast_model()
                self.gradient_list = []

            return leader_pb2.Ack(success=True, message="Gradients received")
    
    def update_and_broadcast_model(self):
        if not self.gradient_list:
            logging.error("No gradients received. Check learner gradient computation.")
            return

        logging.info("Aggregating and applying gradients...")
        num_learners = len(self.gradient_list)
        aggregated_gradients = [torch.zeros_like(param) for param in self.global_model.parameters()]

        # add all gradients
        for learner_grads in self.gradient_list:
            for agg_grad, learner_grad in zip(aggregated_gradients, learner_grads):
                agg_grad.add_(learner_grad.to(self.device))

        # divide by # of learners (this ends the gradient averaging)
        for agg_grad in aggregated_gradients:
            agg_grad.div_(num_learners)

        # set the gradients to the global modal
        for model_param, avg_grad in zip(self.global_model.parameters(), aggregated_gradients):
            model_param.grad = avg_grad

        total_norm = sum(p.grad.norm().item() ** 2 for p in self.global_model.parameters()) ** 0.5
        logging.info(f"Total norm of parameters: {total_norm:.6f}")
        
        # run the optmizer
        self.global_optimizer.step()

        # clear gradients to avoid accumulation
        self.global_optimizer.zero_grad()
        self.broadcast_accumulated_gradient(self.global_model.state_dict())

    def broadcast_accumulated_gradient(self, model_state: dict):
        logging.info("Broadcasting updated model state to learners")
        self.accumulation_count += 1

        # save model to buffer
        model_state_buffer = io.BytesIO()
        torch.save(model_state, model_state_buffer)
        model_state_buffer.seek(0)
        serialized_model_state_dict = model_state_buffer.read()
        
        # if all batches have been trained
        if self.accumulation_count == self.num_batches:
            thread = threading.Thread(target=self.run_model_validation)
            thread.start()
        
        # send model state back to learners for sync
        for learner in self.learner_list:
            serialized_model_state = learner_pb2.ModelState(chunk=serialized_model_state_dict)
            learner['stub'].SyncModelState(serialized_model_state)
    
    def run_model_validation(self):
        logging.info("Starting model validation...")

        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in get_data_loader(valid=True):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del inputs, labels, outputs

        with open('model.pth', 'wb') as f:
            torch.save(self.global_model.state_dict(), f)

        print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * correct / total))

        # program ends
        os._exit(0)

def serve(learner_count):
    logging.info("Starting leader service...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=GRPC_STUB_OPTIONS)
    leader_service = LeaderService(learner_count)
    leader_pb2_grpc.add_LeaderServiceServicer_to_server(leader_service, server)
    
    # run server on the network address
    local_ip = socket.gethostbyname(socket.gethostname())
    server.add_insecure_port(f'{local_ip}:10134')
    server.start()
    logging.info(f'Leader started on {local_ip}:10134')

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Leader server for federated learning")
    parser.add_argument('--learner-count', type=int, required=True, help='Number of learners to wait for before starting training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    serve(args.learner_count)
