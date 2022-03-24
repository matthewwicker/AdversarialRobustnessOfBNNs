# Script to produce posteriors for each dataset under consideration
# We have the hardware to run them two at a time. Remove the ampersands
# to run sequentially
#python3 train.py --dataset kin8nm1 &
#python3 train.py --dataset concrete1
#python3 train.py --dataset boston1 &
#python3 train.py --dataset wine1 &
#python3 train.py --dataset powerplant1 &
#python3 train.py --dataset naval1 &
python3 train.py --dataset energy1 &
python3 train.py --dataset yacht1
