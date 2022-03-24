for eps in 0.005 0.01 0.0125
do
	for (( INNUM=0; INNUM<=250; INNUM++ ))
	do
		python3 estimate_statistical.py --imnum $INNUM --infer VOGN --width 24 --depth 1 --epsilon $eps &
		python3 estimate_statistical.py --imnum $INNUM --infer VOGN --width 24 --depth 2  --epsilon $eps &
		python3 estimate_statistical.py --imnum $INNUM --infer VOGN --width 48 --depth 1  --epsilon $eps &
		python3 estimate_statistical.py --imnum $INNUM --infer VOGN --width 48 --depth 2  --epsilon $eps &
		python3 estimate_statistical.py --imnum $INNUM --infer VOGN --width 64 --depth 1  --epsilon $eps &
		python3 estimate_statistical.py --imnum $INNUM --infer VOGN --width 64 --depth 2  --epsilon $eps
	done
done
