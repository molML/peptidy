python -m pytest peptidy/descriptors.py --doctest-modules --junitxml=tests/results/descriptors.xml 
python -m pytest peptidy/encoding.py --doctest-modules --junitxml=tests/results/encoding.xml 
python -m pytest peptidy/tokenizer.py --doctest-modules --junitxml=tests/results/tokenizer.xml 
python -m pytest peptidy/biology.py --doctest-modules --junitxml=tests/results/biology.xml 

python -m pytest tests/test_encodings_1.py --junitxml=tests/results/encoder_tests_1.xml 
python -m pytest tests/test_encodings_2.py --junitxml=tests/results/encoder_tests_2.xml 

