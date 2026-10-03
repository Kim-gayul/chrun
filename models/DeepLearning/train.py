import torch

# 학습 시작
def train(model, dataloader, loss_fn, optimizer, threshold, device):

    model.train()
    # 평가 지수를 출력하기 위한 변수 정의
    train_loss=0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred = model(batch_x)
        loss = loss_fn(pred, batch_y) 

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # threshold 조정 구간
        pred_prob = torch.sigmoid(pred) 
        predicted = (pred_prob >= threshold).float()
        
        train_loss += loss.item()  # 평균 loss 측정에 사용할 변수
        
        # accuracy측정에 사용할 변수
        train_total += batch_y.size(0)
        train_correct += (predicted.flatten() == batch_y.flatten()).sum().item()

    avg_loss = train_loss / len(dataloader)
    acc = (train_correct / train_total) * 100
    return avg_loss, acc


# 검증 시작
def validate(model, dataloader, loss_fn, threshold, device):

    model.eval() 
    # 평가 지수를 출력하기 위한 변수 정의
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            val_outputs = model(batch_x)
            v_loss = loss_fn(val_outputs, batch_y)

            # threshold 조정 구간
            val_predicted = (torch.sigmoid(val_outputs) >= threshold).float()

            val_loss += v_loss.item()  # 평균 loss 측정에 사용할 변수
            # accuracy측정에 사용할 변수
            val_total += batch_y.size(0)
            val_correct += (val_predicted.flatten() == batch_y.flatten()).sum().item()
              
    avg_loss = val_loss / len(dataloader)
    acc = (val_correct / val_total) * 100 
    return avg_loss, acc