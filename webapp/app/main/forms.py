from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired

class ImageForm(FlaskForm):
    image = FileField(validators=[FileRequired()])