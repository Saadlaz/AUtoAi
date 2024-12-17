from django import forms
from .models import Dataset

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = [ 'file']


class DynamicDatasetForm(forms.Form):
    column_names = forms.CharField(
        label="Column Names (comma-separated)",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Name, Age, Salary'})
    )
    column_types = forms.CharField(
        label="Column Types (comma-separated: int, float, str)",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., str, int, float'})
    )
    num_rows = forms.IntegerField(
        label="Number of Rows",
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 10'})
    )
