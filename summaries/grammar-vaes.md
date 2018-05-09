# Grammar VAEs

Kusner et. al., 2017

[[arxiv]](https://arxiv.org/abs/1703.01925) [[Talk]](https://www.youtube.com/watch?v=ar4Fm1V65Fw)

Authors: Matt J. Kusner, Brooks Paige, José Miguel Hernández-Lobato

### motivation: 
- e.g.: optimising molecules
	- feed in properties and get molecule out, or output local changes to some molecule you've already designed
		- there's a text representation of these molecules (SMILES language)
- key: discrete space of molecules
	- but use continuous latent space as in text (Gómez-Bombarelli et. al., 2016)
		- linear interpolation in latent space gives plausible intermediate sentences, local structure modelled reasonably well
- fit predictor in continuous latent space (latent space learned in an unsupervised space, but turn out to be good features -> surprising)
- BUT challenges of SMILES language:
	- DISADV
		- not natural language, strict formal language, small single-char diffs yield big molecular differences / invalid strings
	- ADV
		- known and fixed underlying grammar
		- context-free
		- no syntactic ambiguity

## Encoding
**One-hot encoding of production rules as opposed to characters**
	- T: seems to be variable-length input depending on length of string representing molecule
- Generate SMILES strings using production rules in the grammar
	- (symbol A can lead to symbol B or C) for each symbol. 
		- Terminal symbols are those that are actually put in the final SMILE string.
	- Linearise parse tree by recording which production rules applied.
	- use sequence of production rules vs sequence of characters
	- use one-hot encodings 

## Decoding
- Only need to eval decoder net once (T: So?)
	- log p(x)s 
- Sampling
	- Mask probability vector to restrict to valid sequences
	- Use a stack. Can record sequence of production rules as well as terminal symbols.

## Data
- Trained on Zinc database of 200k drug-like molecules
- 22 non-terminal symbols in grammar
- up to 120char long (300 prod rules long) per molecule

- Visual latent space is quite smooth.
- Optimise for logP (partition coefficient, marker for how drug-like a molecule is, penalisation for difficulty of synthesis and size of molecule)

## Results
- Produces expressions that are parsed correctly almost all the time for symbols, 31% for molecules 
	- not 100% because there are semantic constraints outside of grammar as well
- LL and RMSE better

## Challenges
- Not everything specified by grammar, e.g. semantic constraints (what bonds are allowed in molecules, runtime errors for programs)

### Toy problems
- Symbolic regression: given function values, what function is this?
	- Results: T: produces nicer interpolations vs character VAE.
- Bayesian optimisation
	- Fit sparse GP to predict output (MSE of function to points)