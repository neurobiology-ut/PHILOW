import os
import time

import torch


def train_model(output_dir, net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    batch_size = dataloaders_dict["train"].batch_size

    # 画像の枚数
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    if dataloaders_dict["val"] is not None:
        num_val_imgs = len(dataloaders_dict["val"].dataset)
        eval_loss_best = 2.0 * num_val_imgs

    # イテレーションカウンタをセット
    iteration = 1

    os.makedirs(output_dir, exist_ok=True)

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                optimizer.zero_grad()
                print('（train）')

            else:
                if dataloaders_dict["val"] is not None:
                    net.eval()  # モデルを検証モードに
                    print('-------------')
                    print('（val）')
                else:
                    continue

            # データローダーからminibatchずつ取り出すループ
            for imges, masks in dataloaders_dict[phase]:
                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)
                masks = masks.to(device)

                # multiple minibatchでのパラメータの更新
                if phase == 'train':
                    optimizer.step()
                    optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(imges)
                    loss = criterion(
                        outputs, masks.float())

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()  # 勾配の計算

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item() / batch_size, duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss += loss.item()

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        if dataloaders_dict["val"] is not None:
            print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
                epoch + 1, epoch_train_loss / num_train_imgs, epoch_val_loss / num_val_imgs))
            print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
            if eval_loss_best > epoch_val_loss:
                torch.save(net.state_dict(), f'{output_dir}/unet_best.pth')
                eval_loss_best = epoch_val_loss
                print('model saved')
            else:
                pass
            val_loss = epoch_val_loss / num_val_imgs
            stop_training = yield epoch + 1, epoch_train_loss / num_train_imgs, val_loss, (imges.cpu().detach().numpy(), masks.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        else:
            val_loss = None
            if dataloaders_dict['test'] is not None:
                for imges, masks in dataloaders_dict['test']:
                    imges = imges.to(device)
                    outputs = net(imges)
                    break
                stop_training = yield epoch + 1, epoch_train_loss / num_train_imgs, val_loss, (imges.cpu().detach().numpy(), None, outputs.cpu().detach().numpy())
            else:
                stop_training = yield epoch + 1, epoch_train_loss / num_train_imgs, val_loss, None
            print('epoch {} || Epoch_TRAIN_Loss:{:.4f}'.format(epoch + 1, epoch_train_loss / num_train_imgs))

        scheduler.step()  # 最適化schedulerの更新
        print(stop_training)
        if stop_training:
            break

    # 最後のネットワークを保存する

    torch.save(net.state_dict(), f'{output_dir}/unet_' + str(epoch + 1) + '.pth')
