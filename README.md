# The review version of the code implementation
This repository is only for the reviewers. The official version will be available to the public upon acceptance.
## ICASSP 2025 (Paper ID 3471)

## Environment
- Recommended: `Python >=3.8`
- Install required python packages. 
    - Refer to `requirements.txt`
    - e.g.) `pip install -r requirements.txt`



## Training Example
```sh
python3 train.py -c configs/config.json -m <run_name>
```

## Sample Page
You can find some speech samples [here](https://lee-jhwn.github.io/temp-icassp25-review/ "speech samples").



## References
#### We adopt some of the backbone code from the following repos:
- https://github.com/lee-jhwn/fesde
- https://github.com/jaywalnut310/vits 
- https://github.com/Uncertain-Quark/s4_eeg
- https://github.com/openspeech-team/openspeech
