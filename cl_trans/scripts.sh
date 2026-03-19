python create_dataset2.py --dataset_type train
python create_dataset2.py --dataset_type valid


python train_lstm.py --accelerator gpu --devices 2 --batch-size 256 --num-workers 2 --num-buy-categories 100 --num-buy-skus 100
python train_dcn.py --accelerator gpu --devices 2 --batch-size 256 --num-workers 2 --num-buy-categories 100 --num-buy-skus 100 --pooling-strategy 'max'
python train_rankmixer.py --accelerator gpu --devices 2 --batch-size 256 --num-workers 2 --num-buy-categories 100 --num-buy-skus 100 --ns-len 5 --hidden-dim 252 --use-semantic True --pooling-strategy 'max'
python train_onetrans.py --accelerator gpu --devices 2 --batch-size 256 --num-workers 2 --num-buy-categories 100 --num-buy-skus 100 --ns-len 6 --num-layers 6 --hidden-dim 256 --final-l-s 20
