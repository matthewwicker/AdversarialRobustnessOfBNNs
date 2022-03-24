# Run all of the statistical estimations in parallel
red=0.60
cls=0
for epsilon in 0.005 0.01 0.0125
do
    for (( INNUM=0; INNUM<=3; INNUM++ ))
    do
	python3 estimate_reduced.py --imnum $INNUM --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $INNUM --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $INNUM --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+11  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+21  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+31  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+41  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+51  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+61  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+71  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+81  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls &
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls &
	I=$(( INNUM+91  ))
	python3 estimate_reduced.py --imnum $I --infer VOGN --epsilon $epsilon --red $red --cls $cls 
	python3 estimate_reduced.py --imnum $I --infer NA --epsilon $epsilon --red $red --cls $cls 
	python3 estimate_reduced.py --imnum $I --infer SWAG --epsilon $epsilon --red $red --cls $cls 
    done
done
