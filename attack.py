from config import parser_handle
parser = parser_handle()
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  

from utils.parameters import My_Params, set_param

param = My_Params()
set_param(param, args)

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import sys

torch.manual_seed(123) 
torch.cuda.manual_seed(123) 

from utils.initial import get_task, applied_attack, get_dataloader
from utils.utils import evaluate

def main():
    # 获取task
    task = get_task(param)
    # 获取对应的综合器
    synthesizer = applied_attack(param, task)
    # 获取对应的数据集
    clean_testset_loader, backdoor_testset_loader, \
                mixed_dataloader = get_dataloader(param, task, synthesizer)

    train_param = {
        "epoch" : 50,
        'model_lr': 1e-3,
        'criterion' : nn.CrossEntropyLoss(),
        'criterion_MSE' : nn.MSELoss(),
    }

    backdoor_model = task.build_model().to(param.device)
    train(backdoor_model, mixed_dataloader, clean_testset_loader, 
          backdoor_testset_loader, param, train_param)
    save_dir = './save'
    os.makedirs(save_dir, exist_ok=True)
    backdoor_model_name = f"{param.dataset}_{param.attack}_backdoor_model.pth"
    backdoor_model_save_path = os.path.join(save_dir, backdoor_model_name)
    torch.save(backdoor_model.state_dict(), backdoor_model_save_path)
    print(f"✅ Model saved successfully: {backdoor_model_save_path}")

def train(model, dataloader, clean_testset_loader, backdoor_testset_loader, param, train_param):
    optimizer = optim.Adam(model.parameters(), lr=train_param['model_lr'])
    criterion = train_param['criterion']
    model.train()
    # 训练模型
    num_epochs = train_param['epoch'] # 增加训练轮数
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", disable=not sys.stdout.isatty())
        
        for data in progress_bar: # for data in mixed_dataloader:
            if len(data) == 3:
                imgs, labels, is_bd = data
            else:
                imgs, labels = data
                
            imgs, labels = imgs.to(param.device), labels.to( param.device)
                
            out = model(imgs)
            loss = criterion(out, labels)
               
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
        
        avg_clean_loss, clean_acc = evaluate(model, clean_testset_loader, criterion,  param.device)
        avg_backdoor_loss, backdoor_acc = evaluate(model, backdoor_testset_loader, criterion,  param.device)
        print("Deep:\nclean_acc:{} , backdoor_acc:{}".format(clean_acc,backdoor_acc))
        print("avg_clean_loss:{} , avg_backdoor_loss:{}\n".format(avg_clean_loss,avg_backdoor_loss))


if __name__ == '__main__':
    main()