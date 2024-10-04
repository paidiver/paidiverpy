document.addEventListener('DOMContentLoaded', loadSchema);

function loadSchema() {
    const schemaUrl = 'configuration-schema.json'; // Replace with the correct path
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
    
    for (const [key, value] of Object.entries(properties)) {
        createFormField(key, value);
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
    if (schema.type === 'string') {
        input = document.createElement('input');
        input.type = 'text';
    } else if (schema.type === 'boolean') {
        input = document.createElement('select');
        const trueOption = document.createElement('option');
        trueOption.value = 'true';
        trueOption.textContent = 'True';
        const falseOption = document.createElement('option');
        falseOption.value = 'false';
        falseOption.textContent = 'False';
        input.appendChild(trueOption);
        input.appendChild(falseOption);
    } else if (schema.type === 'number') {
        input = document.createElement('input');
        input.type = 'number';
    } else if (schema.type === 'array') {
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
