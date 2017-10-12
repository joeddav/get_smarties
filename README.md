# get_smarties
Like `pd.get_dummies`... but smarter.

### The problem

When working with a categorical dataset, most use the 
[`pandas.get_dummies`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
function for easy dummy variable generation. This is well and good, until you have to compare two subsets
of your dataset (as in prediction). If your subsets don't have a row for each possible value for some
feature, your resulting datasets will be different shapes.

For example, say we have a datset with a 'gender' with two possible values: Male and Female.

<table>
<tr><td>id</td><td>...</td><td>gender</td></tr>
<tr><td>1</td><td>...</td><td>Male</td></tr>
<tr><td>2</td><td>...</td><td>Female</td></tr>
<tr><td>3</td><td>...</td><td>Male</td></tr>
</table>

The `pd.get_dummies` function would give you:

<table>
<tr><td>id</td><td>...</td><td>gender_Male</td><td>gender_Female</td></tr>
<tr><td>1</td><td>...</td><td>1</td><td>0</td></tr>
<tr><td>2</td><td>...</td><td>0</td><td>1</td></tr>
<tr><td>3</td><td>...</td><td>1</td><td>0</td></tr>
</table>

But now, say we have another instance and do some machine learning voodoo to predict their gender.
Saw we predict a male. `get_dummies` would give:

<table>
<tr><td>id</td><td>...</td><td>gender_Male</td></tr>
<tr><td>1</td><td>...</td><td>1</td></tr>
</table>

Since Pandas never saw a `Female` in this subset, it only generates a category for `Male`.
The result is that your new and original samples have different shapes, making all kinds
of trouble for computing loss, for example.

See more discussion of this issue at [this thread](https://github.com/pandas-dev/pandas/issues/8918). 

### The solution

`get_smarties` allows you to easily generate dummy variables while persisting the possible values
under each category for you. You can use conventional `fit_transform` and `transform` methods
and solve this problem with no additional effort, like so:

```python
from get_smarties import Smarties
gs = Smarties()

# generate dummies on original dataset, store values for later
X = gs.fit_transform(data)

# generate more dummies on new sample using previously stored values
Y = gs.transform(prediction)
```

And vuala! Your `Y` dataframe will have the same number of columns as `X`.

See short sample notebook at [demo.ipnyb](demo.ipynb).
