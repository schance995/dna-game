# DNA GAME 🧬
Quantum computing's potential is notably dampened by inherent error rates in current quantum hardware, necessitating effective error mitigation strategies to enhance computational accuracy. A promising idea within this domain is to stack multiple Quantum Error Mitigation (QEM) techniques to optimize the performance of quantum computations. 

Our approach diverges from traditional methods by implementing a genetic algorithm to intelligently explore and optimize combinations of error mitigation techniques. By comparing this approach against brute force methods in various simulated environments, we aimed to identify optimal combinations that significantly reduce error rates. Our submission not only demonstrates potential paths to enhanced quantum computation accuracy but also contributes to the ongoing development of the Mitiq toolkit by highlighting areas for improvement.

Developed for [QRISE 2024](https://www.quantumcoalition.io/). Slides [here](https://docs.google.com/presentation/d/1cXJs9C0Gi8Dlbfm8ArY0-wEvWJeQO-lSovG8JE8eypI/view)

## Languages and Tools
DNA Game was developed using the following technologies:
- Mitiq <img src="https://repository-images.githubusercontent.com/236706881/95644100-bd79-11eb-8c37-0aa9d555cb52" width="50px" />
- Cirq <img src="https://quantumai.google/static/site-assets/images/marketing/icons/shared-ic-cirq.png" width="30px" />
- Python <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png" alt="python logo" width="30px"/>

## Setup
Install the dependencies as described in [`pyproject.toml`](pyproject.toml).

Within each run, results are saved to a timestamped folder.

## Notes

Our notes are available in [`QEM_PARAMS.md`](QEM_PARAMS.md) and in [`notebooks/`](notebooks/).

## License

BSD 3-Clause. See [`LICENSE`](LICENSE) for the full text.
