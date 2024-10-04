
// schema = await fetch('configuration-schema.json').then(response => response.json());



document.addEventListener('DOMContentLoaded', loadSchema);

function loadSchema() {
    const schemaUrl = '../../configuration-schema.json';
    fetch(schemaUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load schema');
            }
            return response.json();
        })
        .then(schema => {
            generateForm(schema);
        })
        .catch(error => {
            console.error('Error loading schema:', error);
        });
}

function generateForm(schema) {
    const form = document.getElementById('configForm');
    form.innerHTML = ''; // Clear any existing form elements

    // Assuming schema.properties contains the main configuration sections
    const properties = schema.properties.general.properties;
    console.log(properties);
    console.log(schema.properties.general)
    
    for (const [key, value] of Object.entries(properties)) {
        if (key !== 'sample_data'){
            createFormField(key, value);
        }
    }

    // Show the submit button after form is generated
    document.getElementById('submitButton').style.display = 'block';
}

function createFormField(name, schema) {
    const form = document.getElementById('configForm');
    const formGroup = document.createElement('div');
    formGroup.className = 'form-group';

    const label = document.createElement('label');
    label.textContent = name;
    formGroup.appendChild(label);

    let input;
    let schemaType = schema.type;
    if (typeof(schemaType) === 'object') {
        schemaType = schema.type[0]
    }
    console.log(schemaType)
    if (schemaType === 'string') {
        input = document.createElement('input');
        input.type = 'text';
    } else if (schemaType === 'boolean') {
        input = document.createElement('select');
        const trueOption = document.createElement('option');
        trueOption.value = 'true';
        trueOption.textContent = 'True';
        const falseOption = document.createElement('option');
        falseOption.value = 'false';
        falseOption.textContent = 'False';
        input.appendChild(trueOption);
        input.appendChild(falseOption);
    } else if (schemaType === 'number') {
        input = document.createElement('input');
        input.type = 'number';
    } else if (schemaType === 'array') {
        input = document.createElement('textarea');
        input.placeholder = "Comma separated values";
    }

    if (input) {
        input.name = name;
        formGroup.appendChild(input);
    }

    form.appendChild(formGroup);
}

document.getElementById('submitButton').addEventListener('click', function() {
    const formData = new FormData(document.getElementById('configForm'));
    const config = {};
    for (let pair of formData.entries()) {
        config[pair[0]] = pair[1];
    }
    console.log(config);  // Process or save the config as needed
});
