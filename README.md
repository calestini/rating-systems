# Rating Systems

Implementation of the rating systems to rank sports teams.

## Available

- Elo
	- logistic (logictic decay of score difference)
	- power (logictic decay of score difference)
	- fte (FiveThirtyEight calculation)
- Glicko

## Usage

```python

### initialize rating with parameters
elo = Elo(h=70)
glicko = Glicko(c=42.43, sigma_min=30)
```


## Motivation

## Thoughts
