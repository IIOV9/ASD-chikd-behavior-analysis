from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission here
        name = request.form['Name']
        email = request.form['Email']
        subject = request.form['Subject']
        message = request.form['Message']
        # Process the form data as needed
        print('Received form data:')
        print('Name:', name)
        print('Email:', email)
        print('Subject:', subject)
        print('Message:', message)
        # You can perform further processing or redirect the user after form submission
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
