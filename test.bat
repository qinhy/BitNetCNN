@REM uv run BitNetCNN.py --epochs 1
@REM uv run BitResNet.py --model-size 18 --epochs 1 --epochs 1 --batch-size 64 --dataset-name c100
uv run BitResNet.py --model-size 18 --epochs 1 --epochs 1 --batch-size 64 --dataset-name timnet
uv run BitResNet.py --model-size 50 --epochs 1 --epochs 1 --batch-size 64 --dataset-name c100
uv run BitResNet.py --model-size 50 --epochs 1 --epochs 1 --batch-size 64 --dataset-name timnet
uv run BitMobileNetV4.py --model-size small --epochs 1  --epochs 1 --batch-size 64 --dataset-name c100
uv run BitMobileNetV5.py --model-size tiny  --epochs 1  --epochs 1 --batch-size 64 --dataset-name c100
@REM uv run BitYOLOv8.py --model-size medium --epochs 1
@REM uv run BitConvNeXtv2.py --model-size nano --epochs 1
@REM uv run BitMobileNetV2.py --epochs 1