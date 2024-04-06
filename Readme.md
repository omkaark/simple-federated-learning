# Setup Steps

- pip install -r requirements.txt
- add your model.py file to model_artifacts, define the important variables: device, model, criterion, optimizer_function
- change data loader in utils.py to work with your setup or leave as is and continue using CIFAR-10
- ensure all your computers are on the same network
- start the leader (X is the number of learners you want): python leader.py --learner-count X
- start the learners (ADDRESS:PORT is printed in the leaders logs upon startup): python learner.py --leader-address ADDRESS:PORT
- once X learners join, the training will automatically start, validation accuracy will be printed at the end and model.pth will be generated which is your model binary
