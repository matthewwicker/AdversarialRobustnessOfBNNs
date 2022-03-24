for (( INNUM=0; INNUM<=25; INNUM++ ))
do
	python3 lower.py --imnum $INNUM --infer VOGN --width 24 --depth 1 &
	python3 lower.py --imnum $INNUM --infer VOGN --width 24 --depth 2 &
	python3 lower.py --imnum $INNUM --infer VOGN --width 48 --depth 1 &
	python3 lower.py --imnum $INNUM --infer VOGN --width 48 --depth 2 &
	python3 lower.py --imnum $INNUM --infer VOGN --width 64 --depth 1 &
	python3 lower.py --imnum $INNUM --infer VOGN --width 64 --depth 2 &
#	python3 lower.py --imnum $INNUM --infer SWAG &
#	python3 lower.py --imnum $INNUM --infer NA &
#	python3 lower.py --imnum $INNUM --infer BBB &
done
