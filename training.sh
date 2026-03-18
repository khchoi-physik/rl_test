WIND=${1:-0}
HIDDEN_DIM=${2:-256}
NUM_EPISODES=${3:-1000}


echo "WIND: $WIND"
echo "HIDDEN_DIM: $HIDDEN_DIM"
echo "NUM_EPISODES: $NUM_EPISODES"
echo "Training Lunar Lander"

python train_lander.py "$WIND" "$HIDDEN_DIM" "$NUM_EPISODES"

if [ $? -ne 0 ]; then
    echo "Training failed"
    exit 1
fi

echo "Training completed"
echo "Evaluating Lunar Lander"

python run_lander.py "$WIND" "$HIDDEN_DIM"

if [ $? -ne 0 ]; then
    echo "Evaluation failed"
    exit 1
fi

echo "Evaluation completed"