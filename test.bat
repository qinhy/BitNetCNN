uv run BitNetCNN.py --epochs 1 --batch-size 512

uv run BitResNet.py --epochs 1 --batch-size 64 --dataset-name c100 --model-size 18
uv run BitResNet.py --epochs 1 --batch-size 64 --dataset-name c100 --model-size 50
uv run BitMobileNetV2.py --epochs 1 --batch-size 64 --dataset-name c100 
uv run BitMobileNetV4.py --epochs 1 --batch-size 64 --dataset-name c100 --model-size small
uv run BitMobileNetV5.py --epochs 1 --batch-size 64 --dataset-name c100 --model-size tiny 
uv run BitYOLOv8.py --epochs 1 --batch-size 64 --dataset-name c100 --model-size medium
uv run BitConvNeXtv2.py --epochs 1 --batch-size 64 --dataset-name c100 --model-size nano

uv run BitResNet.py --epochs 1 --batch-size 64 --dataset-name timnet --model-size 18
uv run BitResNet.py --epochs 1 --batch-size 64 --dataset-name timnet --model-size 50
uv run BitMobileNetV2.py --epochs 1 --batch-size 64 --dataset-name timnet
uv run BitMobileNetV4.py --epochs 1 --batch-size 64 --dataset-name timnet --model-size small
uv run BitMobileNetV5.py --epochs 1 --batch-size 64 --dataset-name timnet --model-size tiny 
uv run BitYOLOv8.py --epochs 1 --batch-size 64 --dataset-name timnet --model-size medium
uv run BitConvNeXtv2.py --epochs 1 --batch-size 64 --dataset-name timnet --model-size nano
