pytest --langsmith-output

pytest .\test_smoke.py -vvvs

pytest --basetemp=./.pytest_tmp -vvvs


pytest -q tests/services -vvvs

pytest -q tests/services/test_visrag_service.py