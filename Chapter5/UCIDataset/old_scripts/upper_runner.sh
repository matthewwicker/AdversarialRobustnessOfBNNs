for i in {0..50}
do
    python3 estimate_upper.py --imnum $i --dataset $1 &
done
