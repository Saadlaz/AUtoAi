# core/views.py
import numpy as np
import pandas as pd
import base64
import os
import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.shortcuts import render
from .forms import DatasetForm
from .models import Dataset 
from io import StringIO
from plotly import px 
from io import BytesIO
from sklearn.preprocessing import LabelEncoder , MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from django.http import HttpResponse
from sklearn.impute import SimpleImputer



def upload_dataset(request):
    dataset_info = None

    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['file']
            dataset = Dataset.objects.create(
                name=uploaded_file.name,
                file=uploaded_file
            )

            # Store the dataset ID in the session
            request.session['dataset_id'] = dataset.id

            # Load the dataset into a DataFrame
            file_path = dataset.file.path
            df = pd.read_csv(file_path)

            # Extract dataset info for display
            dataset_info = _get_dataset_info(df)

    else:
        # If a dataset ID exists in the session, pre-load it
        dataset_id = request.session.get('dataset_id')
        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id)
                file_path = dataset.file.path

                # Check if the file exists
                if not os.path.exists(file_path):
                    # Clear the session if the file is missing
                    del request.session['dataset_id']
                    return render(request, 'core/upload.html', {
                        'form': DatasetForm(),
                        'error': "The previously uploaded dataset file is missing. Please upload a new dataset."
                    })

                # Load the dataset into a DataFrame
                df = pd.read_csv(file_path)
                dataset_info = _get_dataset_info(df)
            except Dataset.DoesNotExist:
                # Clear the session if the dataset ID is invalid
                del request.session['dataset_id']
                return render(request, 'core/upload.html', {
                    'form': DatasetForm(),
                    'error': "The dataset associated with the session does not exist. Please upload a new dataset."
                })

    return render(request, 'core/upload.html', {'form': DatasetForm(), 'dataset_info': dataset_info})


def _get_dataset_info(df):
    """Helper function to extract dataset information."""
    buffer = StringIO()
    df.info(buf=buffer)
    info_output = buffer.getvalue()
    buffer.close()

    column_info = []
    for line in info_output.splitlines()[5:]:
        if "dtypes:" in line or "memory usage:" in line:
            continue
        parts = line.split(maxsplit=4)
        if len(parts) == 5:
            column_info.append({
                'Index': parts[0],
                'ColumnName': parts[1],
                'NonNullCount': parts[2],
                'DataType': parts[4],
            })

    return {
        'preview': df.head(15).to_html(classes='table table-striped table-hover', index=False),
        'info': column_info,
        'describe': df.describe().reset_index().to_html(classes='table table-striped table-hover', index=False),
        'shape': df.shape,
    }



def encode_data(request):
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return render(request, 'core/upload.html', {'error': 'No dataset available for encoding. Please upload a dataset first.'})

    # Fetch and load the dataset
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file.path
        df = pd.read_csv(file_path)
    except Exception as e:
        return render(request, 'core/upload.html', {'error': f"Error loading dataset: {str(e)}"})

    # Store mappings for each categorical column
    label_mappings = {}

    # Encode categorical features
    try:
        categorical_columns = df.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            label_encoder = LabelEncoder()
            for column in categorical_columns:
                # Store the mapping
                label_mappings[column] = dict(zip(label_encoder.fit_transform(df[column]), df[column]))
                # Encode the feature
                df[column] = label_encoder.transform(df[column])

            # Save the encoded data
            df.to_csv(file_path, index=False)

            # Save the mapping to the session (or a database if needed)
            request.session['label_mappings'] = label_mappings
        else:
            return render(request, 'core/upload.html', {'error': 'No categorical columns to encode.'})
    except Exception as e:
        return render(request, 'core/upload.html', {'error': f"Error encoding dataset: {str(e)}"})

    return render(request, 'core/upload.html', {'message': 'Dataset successfully encoded and saved.'})


def _plot_to_base64(fig):
    """Convert Matplotlib figure to a Base64 image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return base64_image

def visualize_data(request):

    # Retrieve dataset ID from session
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return render(request, 'core/visualize.html', {'error': 'No dataset available for visualization. Please upload a dataset first.'})

    # Load dataset
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        df = pd.read_csv(dataset.file.path)
    except Exception as e:
        return render(request, 'core/visualize.html', {'error': f"Error loading dataset: {str(e)}"})

    # Encode categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

    # Get user input for plot type and labels
    plot_type = request.GET.get('plot_type', 'Heatmap')  # Default to Heatmap
    x_label = request.GET.get('x_label')
    y_label = request.GET.get('y_label')

    plot = None

    try:
        if plot_type == 'Violin Plot':
            x_feature = request.GET.get('x_feature')
            y_feature = request.GET.get('y_feature')

            if x_feature and y_feature:
                if x_feature in df.columns and y_feature in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.violinplot(x=df[x_feature], y=df[y_feature], palette="muted", ax=ax)
                    ax.set_title(f"Violin Plot of {y_feature} by {x_feature}", fontsize=14)
                    ax.set_xlabel(x_feature)
                    ax.set_ylabel(y_feature)
                    plot = _plot_to_base64(fig)
                else:
                    return render(request, 'core/visualize.html', {
                        'error': f"Selected features {x_feature} or {y_feature} not found in dataset."
                    })
            else:
                return render(request, 'core/visualize.html', {
                    'error': "Please select both X and Y features for the violin plot."
                })

        elif plot_type == 'Heatmap':
            
            excluded_columns = ['Id']  # Add other columns to exclude if necessary
            filtered_df = df.drop(columns=excluded_columns, errors='ignore')
            
            # Generate the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                filtered_df.corr(),
                annot=True,          
                fmt=".2f",           
                cmap='coolwarm',     
                cbar=True,           
                linewidths=0.5,      
                ax=ax
            )
            ax.set_title("Correlation Heatmap (With Correlation Values)", fontsize=14)
            plot = _plot_to_base64(fig)



        elif plot_type == 'Histogram' and x_label:
            # Histogram for selected feature
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[x_label], kde=True, ax=ax)
            ax.set_title(f"Histogram of {x_label}", fontsize=14)
            plot = _plot_to_base64(fig)

        elif plot_type == 'Scatter' and x_label and y_label:
            # Scatter plot for selected X and Y labels
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=df[x_label], y=df[y_label], ax=ax)
            ax.set_title(f"Scatter Plot: {x_label} vs {y_label}", fontsize=14)
            plot = _plot_to_base64(fig)

        elif plot_type == 'Pair Plot':
            # Pair plot of all features
            fig = sns.pairplot(df)
            plot = _plot_to_base64(fig.fig)

        elif plot_type == 'Pie' and x_label:
            # Pie chart for selected feature
            fig, ax = plt.subplots(figsize=(8, 6))
            df[x_label].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90, cmap='viridis')
            ax.set_ylabel('')
            ax.set_title(f"Pie Chart for {x_label}", fontsize=14)
            plot = _plot_to_base64(fig)
      
    except Exception as e:
        return render(request, 'core/visualize.html', {'error': f"Error generating plot: {str(e)}"})

    # Pass plot and available columns to the template
    return render(request, 'core/visualize.html', {
        'plot': plot,
        'plot_type': plot_type,
        'columns': df.columns.tolist(),
        'x_label': x_label,
        'y_label': y_label,
    })

def home(request):
    return render(request, 'core/home.html')

def upload(request):
    return render(request, 'core/upload.html')

def visualize(request):
    return render(request, 'core/visualize.html')

def preprocess(request):
    return render(request, 'core/preprocess.html')

def classify(request):
    return render(request, 'core/classify.html')

def predict(request):
    return render(request, 'core/predict.html')


#-------------------------------------------------------


def preprocess(request):
    df = None
    missing_summary, train_shape, test_shape, sample_data = None, None, None, None
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return render(request, 'core/upload.html', {'error': 'No dataset available for encoding. Please upload a dataset first.'})

    try:
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file.path
        df = pd.read_csv(file_path)
    except Exception as e:
        return render(request, 'core/upload.html', {'error': f"Error loading dataset: {str(e)}"})

    # Process the dataset if available
    if df is not None:
        # Handle Missing Values
        missing_summary = df.isnull().sum().to_dict()
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype == 'object':  # Categorical columns
                    df[column].fillna("Unknown", inplace=True)
                else:  # Numerical columns
                    df[column].fillna(df[column].median(), inplace=True)

        # Remove Duplicates
        df.drop_duplicates(inplace=True)

        # Strip Whitespaces
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Encoding Categorical Features
        for col in df.select_dtypes(include=['object']).columns:
            # High cardinality -> Label Encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])         

        # Outlier Detection
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            upper_limit = df[col].quantile(0.95)
            lower_limit = df[col].quantile(0.05)
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

        # Scaling & Normalization
        scaler = MinMaxScaler()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Train-Test Split
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train_shape, test_shape = train.shape, test.shape

        # Prepare Data Preview
        sample_data = df.head(10).to_html(classes="table table-bordered")

    # Context for rendering
    context = {
        'missing_summary': missing_summary,
        'train_shape': train_shape,
        'test_shape': test_shape,
        'sample_data': sample_data,
    }
    return render(request, 'core/preprocess.html', context)


# Preprocessing utility functions
def fill_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    numeric_data = data.select_dtypes(include=[np.number])
    data[numeric_data.columns] = imputer.fit_transform(numeric_data)
    return data

def scale_data(data):
    scaler = StandardScaler()
    numeric_data = data.select_dtypes(include=[np.number])
    data[numeric_data.columns] = scaler.fit_transform(numeric_data)
    return data

def remove_outliers(data):
    from scipy.stats import zscore
    numeric_data = data.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_data))
    data = data[(z_scores < 3).all(axis=1)]  # Retain rows where Z-score < 3
    return data