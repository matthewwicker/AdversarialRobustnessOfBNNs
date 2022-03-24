# Author: Matthew Wicker

for (( INNUM=0; INNUM<=10; INNUM++ ))
do
	python3 estimate_lower.py --imnum $IMNUM --infer VOGN &
	python3 estimate_lower.py --imnum $IMNUM --infer NA &
	python3 estimate_lower.py --imnum $IMNUM --infer SWAG &
	I=$(( INNUM+11  ))
        python3 estimate_lower.py --imnum $I --infer VOGN &
        python3 estimate_lower.py --imnum $I --infer NA &
        python3 estimate_lower.py --imnum $I --infer SWAG &
        I=$(( INNUM+21  ))
        python3 estimate_lower.py --imnum $I --infer VOGN &
        python3 estimate_lower.py --imnum $I --infer NA &
        python3 estimate_lower.py --imnum $I --infer SWAG &
        I=$(( INNUM+31  ))
        python3 estimate_lower.py --imnum $I --infer VOGN &
        python3 estimate_lower.py --imnum $I --infer NA &
        python3 estimate_lower.py --imnum $I --infer SWAG &
        I=$(( INNUM+41  ))
        python3 estimate_lower.py --imnum $I --infer VOGN &
        python3 estimate_lower.py --imnum $I --infer NA &
        python3 estimate_lower.py --imnum $I --infer SWAG 
done
