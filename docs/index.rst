.. UGNN documentation master file, created by
   sphinx-quickstart on Wed Apr 16 16:00:30 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

UGNN Documentation
===================

Welcome to the documentation for **UGNN**, a library for the using the **unfolded graph neural network** (UGNN) model for discrete-time dynamic graphs.

For more details on this model see the paper, `Valid Conformal Prediction for Dynamic GNNs <https://arxiv.org/abs/2405.19230>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   modules

Installation
------------

To install UGNN, use the following command:

.. code-block:: bash

   pip install ugnn

.. Quick Start
.. -----------

.. Here’s a quick example of how to use UGNN:

.. .. code-block:: python

..    from ugnn import conformal

..    # Example usage
..    result = conformal.get_prediction_sets(output, data, calib_mask, test_mask)
..    print(result)

Additional Resources
---------------------

- `GitHub Repository <https://github.com/edwarddavis1/UGNN>`_
- `Issue Tracker <https://github.com/edwarddavis1/UGNN/issues>`_