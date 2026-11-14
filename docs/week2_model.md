Week 2 model report

Owner: Teammate 2.

Existing starter_core.py source and license record

* Reused the existing starter_core.py implementation.
* Kept the original source code unchanged.
* Imported the existing Batch and make_model implementations.
* Original source attribution was retained.
* License record: The original repository license and source header should be checked before final submission.

Model dimensions, dropout and parameter counts

* Number of layers: 1
* Model dimension (d_model): 128
* Feed-forward dimension (d_ff): 256
* Dropout: 0.1
* Attention heads compared: 1 and 8
* Head dimension with 1 head: 128
* Head dimension with 8 heads: 16
* Parameters with 1 head: 337,936
* Parameters with 8 heads: 337,936

Both models have the same parameter count and identical initial weights when initialized with the same random seed (11).

One-head and eight-head output shapes

Using the fixture configuration and a batch of two examples:

Output	1 head	8 heads
Hidden states	(2, 3, 128)	(2, 3, 128)
Generator scores	(2, 3, 16)	(2, 3, 16)

The generator output probabilities sum to approximately 1 for each position.

Causal mask test result

* test_causal_mask: PASSED
* Changing a future target token does not change earlier decoder outputs.

Decoder EOS and length-limit test results

All decoder tests passed:

* test_immediate_eos: PASSED
* test_token_then_eos: PASSED
* test_output_limit: PASSED
* test_batch_rows_finish_at_different_steps: PASSED

The decoder handles immediate EOS, normal EOS termination, maximum generation length, and different EOS positions across batch rows.

The default maximum generation length is 128 tokens.

Tiny synthetic learning check

* Not performed.
* This is an optional check and was not included in the six required unit tests.

Commands run and observed outputs

Command 1:

python -m week2.model --check

Observed output:

heads 1 hidden (2, 3, 128) parameters 337936
heads 8 hidden (2, 3, 128) parameters 337936

Command 2:

python -m unittest discover -s tests -p "test_week2_model.py" -v

Observed output:

Ran 6 tests in 0.054s
OK

All six unit tests passed.

Known limitations

* The optional tiny synthetic learning check has not been performed.
* No real training or translation quality evaluation was performed as part of these unit tests.
* The decoder uses greedy decoding.
* The original repository license record should be verified before final submission.

No peer-review sign-off is required in Week 2.
