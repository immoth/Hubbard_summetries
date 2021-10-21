
"""A sample runtime program that submits random circuits for user-specified iterations."""
import random
from qiskit import transpile
from qiskit.circuit.random import random_circuit



def prepare_circuits(backend):
    """Generate a random circuit.

    Args:
        backend: Backend used for transpilation.

    Returns:
        Generated circuit.
    """
    circuit = random_circuit(num_qubits=5, depth=4, measure=True,
                             seed=random.randint(0, 1000))
    return transpile(circuit, backend)


def main(backend, user_messenger, **kwargs):
    """Main entry point of the program.

    Args:
        backend: Backend to submit the circuits to.
        user_messenger: Used to communicate with the program consumer.
        kwargs: User inputs.
    """
    iterations = kwargs.pop('iterations', 5)
    for it in range(iterations):
        qc = prepare_circuits(backend)
        result = backend.run(qc).result()
        user_messenger.publish({"iteration": it, "counts": result.get_counts()})

    return "All done!"
