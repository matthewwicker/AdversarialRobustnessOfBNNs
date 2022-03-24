# Run all of the statistical estimations in parallel
for (( INNUM=27; INNUM<=100; INNUM++ ))
do
	python3 estimate_statistical.py --imnum $INNUM --infer HMC &
	python3 estimate_statistical.py --imnum $INNUM --infer BBB &
	python3 estimate_statistical.py --imnum $INNUM --infer VOGN &
	python3 estimate_statistical.py --imnum $INNUM --infer NA &
	python3 estimate_statistical.py --imnum $INNUM --infer SWAG
done
