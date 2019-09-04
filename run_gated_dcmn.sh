for nhop in 1 2 3 4 5 6 7 8 9
do
  for lr in 0.001 0.01 0.1
  do
    for fold in 0 1 2 3 4
    do
      echo "lr:${lr} nhop:${nhop} fold:${fold} optimizer: sgd"
      set PYTHONHASHSEED=1
      python train_gated_dcmn.py --network gated_dcmn --setting ecg-clinical-matched --lr ${lr} --nhop ${nhop} --name normedclassweight --edim 50 --batchsize 32 --gpu 0 --optimizer sgd --fold ${fold}  >gated_dcmn_logs/gated_dcmn_lstm_lr:${lr}nhop:${nhop}fold:${fold}_sgd.log
    done
  done
done
