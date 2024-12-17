# core/views.py
import pandas as pd
from django.shortcuts import render
from .forms import DatasetForm
from .models import Dataset 
from io import StringIO
import os
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from django.http import HttpResponse
from sklearn.impute import SimpleImputer
import numpy as np
import plotly.express as px




def upload_dataset(request):
    dataset_info = {}  # Dictionary to hold dataset details

    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['file']

            # Save uploaded file
            dataset = Dataset.objects.create(
                name=f"dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                file=uploaded_file
            )

            # Save dataset ID in session
            request.session['dataset_id'] = dataset.id

            # Load dataset and get information
            file_path = dataset.file.path
            df = pd.read_csv(file_path)

            # Use the helper function to get dataset information
            dataset_info = _get_dataset_info(df)

    else:
        # Handle dataset reload if ID exists in session
        form = DatasetForm()
        dataset_id = request.session.get('dataset_id')
        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id)
                file_path = dataset.file.path
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    dataset_info = _get_dataset_info(df)
            except Dataset.DoesNotExist:
                # Clear session if dataset doesn't exist
                del request.session['dataset_id']

    return render(request, 'core/upload.html', {'form': form, 'dataset_info': dataset_info})


def _get_dataset_info(df):
    """Helper function to extract dataset information."""
    buffer = StringIO()
    df.info(buf=buffer)
    info_output = buffer.getvalue()
    buffer.close()

    # Process Pandas DataFrame info output
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
        'preview': df.head(10).to_html(classes='table table-striped table-hover', index=False),
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



def visualize_data(request):
    # Load dataset
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return render(request, 'core/visualize.html', {'error': 'No dataset available. Please upload one.'})

    try:
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file.path
        df = pd.read_csv(file_path)
    except Exception as e:
        return render(request, 'core/visualize.html', {'error': f"Error loading dataset: {str(e)}"})

    # Extract column names
    columns = df.columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Get user inputs
    plot_type = request.GET.get('plot_type', 'Heatmap')
    x_label = request.GET.get('x_label', None)
    y_label = request.GET.get('y_label', None)

    plot_html = None

    try:
        # If Heatmap is selected, encode categorical data
        if plot_type == 'Heatmap':
            if categorical_columns:
                label_encoder = LabelEncoder()
                for column in categorical_columns:
                    df[column] = label_encoder.fit_transform(df[column])

            # Generate the Heatmap with a custom color scale and size
            corr = df.corr()
            fig = px.imshow(
                corr, 
                text_auto=True, 
                title="Correlation Heatmap (Encoded Data)", 
                width=1200,  # Width
                height=900,  # Height
                color_continuous_scale='Viridis'  # Use a predefined color scale
            )
            plot_html = fig.to_html(full_html=False)


        elif plot_type == 'Histogram' and x_label:
            fig = px.histogram(df, x=x_label, title=f"Histogram of {x_label}")
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Scatter' and x_label and y_label:
            fig = px.scatter(df, x=x_label, y=y_label, title=f"Scatter Plot: {x_label} vs {y_label}")
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Pair Plot':
            fig = px.scatter_matrix(df, dimensions=numerical_columns, title="Pair Plot of Numerical Features")
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Pie' and x_label:
            fig = px.pie(df, names=x_label, title=f"Pie Chart of {x_label}")
            plot_html = fig.to_html(full_html=False)

        elif plot_type == 'Violin Plot' and x_label and y_label:
            fig = px.violin(df, x=x_label, y=y_label, box=True, title=f"Violin Plot of {y_label} by {x_label}")
            plot_html = fig.to_html(full_html=False)

    except Exception as e:
        return render(request, 'core/visualize.html', {'error': f"Error generating plot: {str(e)}"})

    return render(request, 'core/visualize.html', {
        'plot_html': plot_html,
        'columns': columns,
        'plot_type': plot_type,
        'x_label': x_label,
        'y_label': y_label,
    })





#-------------------------------------------------------



from django.shortcuts import render, redirect
from django.conf import settings
from .forms import DynamicDatasetForm
from .models import Dataset
import pandas as pd
import numpy as np
import os
from django.utils import timezone

def create_dynamic_dataset(request):
    if request.method == 'POST':
        form = DynamicDatasetForm(request.POST)
        if form.is_valid():
            column_names = [name.strip() for name in form.cleaned_data['column_names'].split(',')]
            column_types = [typ.strip().lower() for typ in form.cleaned_data['column_types'].split(',')]
            num_rows = form.cleaned_data['num_rows']

            # Generate Data Dynamically
            data = {}
            for col_name, col_type in zip(column_names, column_types):
                if col_type == 'int':
                    data[col_name] = np.random.randint(0, 100, num_rows)
                elif col_type == 'float':
                    data[col_name] = np.random.uniform(0, 100, num_rows)
                elif col_type == 'str':
                    data[col_name] = [f"{col_name}_{i}" for i in range(num_rows)]
                else:
                    data[col_name] = ['N/A'] * num_rows  # Default to N/A for unknown types
            
            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Save CSV File
            file_name = f"dynamic_dataset_{timezone.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            df.to_csv(file_path, index=False)

            # Save to Database
            Dataset.objects.create(
                name=file_name,
                file=file_name
            )

            # Redirect to upload page
            return redirect('upload_dataset')
    else:
        form = DynamicDatasetForm()

    return render(request, 'core/create_dynamic_dataset.html', {'form': form})



def home(request):
    return render(request, 'core/home.html')

def upload(request):
    return render(request, 'core/upload.html')



def preprocess(request):
    return render(request, 'core/preprocess.html')

def classify(request):
    return render(request, 'core/classify.html')

def predict(request):
    return render(request, 'core/predict.html')


#-------------------------------------------------------


def preprocess(request):
    """
    Render the preprocess.html template and display dataset info if available.
    """
    dataset_id = request.session.get('dataset_id')
    dataset_info = None

    if dataset_id:
        try:
            # Retrieve the dataset and load it into a DataFrame
            dataset = Dataset.objects.get(id=dataset_id)
            file_path = dataset.file.path
            df = pd.read_csv(file_path)

            # Extract dataset info for display
            dataset_info = {
                'name': dataset.name,
                'rows': df.shape[0],
                'columns': df.shape[1],
            }
        except Dataset.DoesNotExist:
            return HttpResponse("The dataset associated with the session does not exist.", status=400)
        except Exception as e:
            return HttpResponse(f"Error loading dataset: {str(e)}", status=400)

    return render(request, 'core/preprocess.html', {'dataset_info': dataset_info})


def submit_preprocessing(request):
    """
    Handle preprocessing logic using the previously uploaded dataset
    and display the results on the page.
    """
    if request.method == 'POST':
        # Retrieve the dataset ID from the session
        dataset_id = request.session.get('dataset_id')
        if not dataset_id:
            return HttpResponse("No dataset found in session. Please upload a dataset first.", status=400)

        try:
            # Get the dataset object and load it into a DataFrame
            dataset = Dataset.objects.get(id=dataset_id)
            file_path = dataset.file.path
            df = pd.read_csv(file_path)
        except Dataset.DoesNotExist:
            return HttpResponse("The dataset associated with the session does not exist.", status=400)
        except Exception as e:
            return HttpResponse(f"Error loading dataset: {str(e)}", status=400)

        # Initialize a dictionary to store preprocessing actions and stats
        processing_info = {
            "original_shape": df.shape,
            "actions": [],
        }

        # Apply preprocessing steps based on user selection
        if 'fill_missing' in request.POST:
            df = fill_missing_values(df)
            processing_info["actions"].append("Filled missing values")

        if 'scale_data' in request.POST:
            df = scale_data(df)
            processing_info["actions"].append("Scaled numeric data")

        if 'remove_outliers' in request.POST:
            original_rows = df.shape[0]
            df = remove_outliers(df)
            removed_rows = original_rows - df.shape[0]
            processing_info["actions"].append(f"Removed {removed_rows} outliers")

        # Save processed DataFrame to memory for display
        processed_data_preview = df.head(10).to_html(classes="table table-striped", index=False)

        # Update the processing information
        processing_info["processed_shape"] = df.shape

        return render(request, 'core/preprocess.html', {
            "dataset_info": {
                "name": dataset.name,
                "rows": processing_info["processed_shape"][0],
                "columns": processing_info["processed_shape"][1],
            },
            "processing_info": processing_info,
            "processed_data_preview": processed_data_preview,
        })

    return HttpResponse("Invalid request method.", status=405)




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