import importlib
import argparse
import io
import socket
import threading
import grpc
from concurrent import futures
import time
import torch
import logging
from protos import leader_pb2, leader_pb2_grpc
from protos import learner_pb2, learner_pb2_grpc
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GRPC_STUB_OPTIONS = [
    ('grpc.max_send_message_length', 50 * 1024 * 1024),  # For example, 50 MB
    ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # For example, 50 MB
]

class LearnerService(learner_pb2_grpc.LearnerServiceServicer):
    def __init__(self, network_addr, leader_stub, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None
        self.model = None
        self.criterion = None
        self.leader_stub = leader_stub
        self.network_addr = network_addr
        self.data_batches = []
        self.sync_model_event = threading.Event()

    def load_model(self):
        logging.info('Getting model...')
        model_stream = self.leader_stub.GetModel(leader_pb2.Empty())

        # define the directory and model path
        model_dir = './learner_model_artifacts'
        model_path = os.path.join(model_dir, 'model.py')

        # ensure the directory exists
        os.makedirs(model_dir, exist_ok=True)

        # write the model data to model.py
        with open(model_path, 'wb') as model_file:
            for model_data in model_stream:
                model_file.write(model_data.chunk)

        # dynamically import file
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # load relevant variables from the file
        self.device = model_module.device
        self.model = model_module.model.to(self.device)
        self.criterion = model_module.criterion

        logging.info('Model loaded successfully and moved to device')

    def training_loop(self):
        logging.info('Starting training loop')

        # get batches
        self.get_data_batches()

        # for each batch in the set of received batches
        for batch in self.data_batches:
            logging.info('Training batch...')

            # unpack batch
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            first_number = inputs.view(-1)[0].item()
            print(f"First number in input tensor: {first_number}")

            # forward pass
            outputs = self.model(inputs)

            # back prop
            loss = self.criterion(outputs, labels)
            loss.backward()
            logging.info(f'Loss: {loss.item():.6f}')

            # unpack and send gradients
            gradients = [param.grad for param in self.model.parameters() if param.grad is not None]

            print(f'Gradient norms: {gradients[0].norm().item()}')

            # send gradients to leader
            buffer = io.BytesIO()
            torch.save(gradients, buffer)
            buffer.seek(0)
            serialized_gradients = buffer.read()
            logging.info('Sending gradients...')
            self.leader_stub.AccumulateGradients(leader_pb2.GradientData(chunk=serialized_gradients))

            # clear gradients to avoid accumulation
            self.model.zero_grad()

            # wait for all learners to send gradients and receive the new model
            self.sync_model_event.wait()
            self.sync_model_event.clear()
        
        logging.info('Completed one training loop iteration')

        self.training_loop()
    
    def StartTraining(self, request, context):
        logging.info('Received start training signal from leader')
        thread = threading.Thread(target=self.training_loop)
        thread.start()
        return learner_pb2.Ack(success=True, message="Model training started")

    def get_data_batches(self):
        logging.info('Getting data batches...')
        data_req = leader_pb2.LearnerDataRequest(network_addr=self.network_addr)
        data_stream = self.leader_stub.GetData(data_req)
        batches = []
        data_received = False

        # each chunk has one batch
        for data_chunk in data_stream:
            buffer = io.BytesIO(data_chunk.chunk)
            try:
                tensor = torch.load(buffer)
            except Exception as e:
                logging.error(f'Error loading data batch: {e}')
            finally:
                batches.append(tensor)
                data_received = True

        if not data_received: # If no data is received, the learner is done training
            logging.info('Model is done training, exiting...')
            os._exit(0)

        self.data_batches = batches

    def SyncModelState(self, request, context):
        logging.info('Synchronizing model state...')

        # load model
        buffer = io.BytesIO(request.chunk)
        new_model_state = torch.load(buffer, map_location=self.device)
        self.model.load_state_dict(new_model_state)
        self.sync_model_event.set()

        first_param = next(self.model.parameters()).detach().cpu().numpy().flatten()[0]
        logging.info(f'First parameter after sync: {first_param}')
        logging.info('Model state synchronized successfully')
        
        return learner_pb2.Ack(success=True, message="Model state synchronized successfully")

def serve(network_addr, learner_port, leader_stub):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=GRPC_STUB_OPTIONS)
    learner_service = LearnerService(network_addr, leader_stub)
    learner_pb2_grpc.add_LearnerServiceServicer_to_server(learner_service, server)
    learner_service.load_model()
    logging.info(f'Learner started on {network_addr}')
    server.add_insecure_port(f'0.0.0.0:{learner_port}')
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Learner Service')
    parser.add_argument('--leader-address', type=str, required=True, help='Network address of the leader in the form 192.168.xxx.xxx:xxxx')
    parser.add_argument('--port', type=int, default=10135, help='Port you want the learner to use')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    leader_address = args.leader_address
    learner_port = args.port

    channel = grpc.insecure_channel(leader_address, options=GRPC_STUB_OPTIONS)
    leader_stub = leader_pb2_grpc.LeaderServiceStub(channel)

    network_addr = f"{socket.gethostbyname(socket.gethostname())}:{learner_port}"
    learner_info = leader_pb2.LearnerInfo(network_addr=network_addr)

    logging.info('Registering learner...')
    is_registered = leader_stub.RegisterLearner(learner_info)
    if is_registered.success:
        serve(network_addr, learner_port, leader_stub)
    else:
        logging.error('Registering learner unsuccessful')
