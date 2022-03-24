for i in {0..10}
do
    python3 estimate_upper.py --imnum $i --dataset boston1 &
    python3 estimate_upper.py --imnum $i --dataset energy1 &
    python3 estimate_upper.py --imnum $i --dataset yacht1 &
    python3 estimate_upper.py --imnum $i --dataset kin8nm1 &
    python3 estimate_upper.py --imnum $i --dataset powerplant1 &
    python3 estimate_upper.py --imnum $i --dataset wine1 &
    python3 estimate_upper.py --imnum $i --dataset naval1 &
    python3 estimate_upper.py --imnum $i --dataset concrete1 
done
