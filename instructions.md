Yes, annotations are generated in the notebooks/quickstart.ipynb notebook.
Here's how and where annotation happens:
In the final evaluation step (the last code cell), the function evaluate_hypotheses is called. This function takes the natural language hypotheses (generated from interpreting neurons) and applies them to a holdout set of texts to determine if each hypothesis is present in each text.
The output includes a line:
This means the notebook is generating (annotating) labels for each hypothesis on the holdout set, i.e., for each hypothesis, it checks whether the concept described by the hypothesis is present in each review.
So, yes: the notebook does generate annotations, specifically in the context of evaluating how well the hypotheses generalize to new, unseen data. This is done automatically as part of the evaluation process.
what about this stage?

Great question! At this stage in the notebook—where you call generate_hypotheses—annotations are not being generated for the holdout set, but annotations are being generated for the training set (or a subset of it) as part of the hypothesis interpretation and fidelity scoring process.
What happens in this stage?
generate_hypotheses selects the most predictive neurons, interprets them (generates natural language hypotheses), and then scores the fidelity of these interpretations.
Fidelity scoring means: for each neuron and its interpretation, the code checks how well the neuron's activation matches the presence/absence of the concept described by the interpretation in the training texts.
To do this, the code needs to annotate (label) whether each training text contains the concept described by the hypothesis. This is done using an LLM or a rule-based annotator, depending on the implementation.
You can see this in the notebook output:
This means the code is generating (annotating) labels for the training set to evaluate how well the interpretation matches the neuron's activation.
Summary
Yes, annotations are generated at this stage, but only for the training set (not the holdout set).
These annotations are used to compute the "fidelity score" for each hypothesis, which measures how well the neuron's activation aligns with the natural language interpretation.
The holdout set annotations are generated later, during the final evaluation step.
but how is then correlation calculated?

Excellent question! The calculation of correlation in the generate_hypotheses stage does not depend on LLM-generated annotations. Here’s how it works:
How is correlation calculated in this context?
The correlation (specifically, Pearson correlation) is computed between:
The activation values of each neuron (from the SAE) for each training example.
The target variable (e.g., the stars rating for each review).
Step-by-step:
For each neuron:
You have a vector of activations: one value per training example.
You have a vector of labels: the target value (e.g., star rating) for each training example.
Pearson correlation is computed between these two vectors.
This gives a measure of how strongly the neuron's activation is linearly related to the target variable.
No LLM annotation is needed for this step!
The correlation is purely a statistical calculation on the numeric activations and the numeric labels.
The LLM-based annotation is only used for scoring the interpretation fidelity (i.e., how well the natural language description matches the neuron's activation), not for the correlation calculation itself.
In summary:
Correlation is calculated directly from neuron activations and the target labels.
Annotations (LLM or otherwise) are not used for the correlation calculation—they are only used for interpretation fidelity and for evaluating hypotheses on the holdout set.
If you want to see the code for this, it will be in the function that implements the "correlation" selection method, likely in select_neurons.py. Would you like to see the exact code for how this is done?