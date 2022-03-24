# Run all of the statistical estimations in parallel
for epsilon in 0.025 0.075 0.1 0.125
do
    for (( INNUM=0; INNUM<=3; INNUM++ ))
    do
	python3 estimate_statistical.py --imnum $INNUM --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $INNUM --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $INNUM --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+11  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+21  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+31  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+41  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+51  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+61  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+71  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+81  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon &
	I=$(( INNUM+91  ))
	python3 estimate_statistical.py --imnum $I --infer VOGN --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer NA --epsilon $epsilon &
	python3 estimate_statistical.py --imnum $I --infer SWAG --epsilon $epsilon 
    done
done

