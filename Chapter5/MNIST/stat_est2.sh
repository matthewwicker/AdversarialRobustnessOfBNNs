# Run all of the statistical estimations in parallel
for (( INNUM=0; INNUM<=10; INNUM++ ))
do
	python3 estimate_statistical.py --imnum $INNUM --infer VOGN &
	python3 estimate_statistical.py --imnum $INNUM --infer NA &
	python3 estimate_statistical.py --imnum $INNUM --infer SWAG &
	python3 estimate_statistical.py --imnum $INNUM --infer HMC &
	python3 estimate_statistical.py --imnum $INNUM --infer BBB &
	I=$(( INNUM+11  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer SWAG &
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB &
	I=$(( INNUM+21  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB &
	python3 estimate_statistical.py --imnum $I --infer SWAG &
	I=$(( INNUM+31  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer SWAG &
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB &
	I=$(( INNUM+41  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer SWAG &
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB &
	I=$(( INNUM+51  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer SWAG &
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB &
	I=$(( INNUM+61  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer SWAG &
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB &
	I=$(( INNUM+71  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer SWAG 
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB &
	I=$(( INNUM+81  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer SWAG 
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB &
	I=$(( INNUM+91  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN &
	python3 estimate_statistical.py --imnum $I --infer NA &
	python3 estimate_statistical.py --imnum $I --infer SWAG &
	python3 estimate_statistical.py --imnum $I --infer HMC &
	python3 estimate_statistical.py --imnum $I --infer BBB 
done
