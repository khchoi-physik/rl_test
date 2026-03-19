WIND=0
HIDDEN_DIM=512
NUM_EPISODES=3000
NUM_SIMULATIONS=4
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
python run_lander.py "$WIND" "$HIDDEN_DIM" "$NUM_SIMULATIONS"
if [ $? -ne 0 ]; then
    echo "Evaluation failed"
    exit 1
fi
echo "Evaluation completed"