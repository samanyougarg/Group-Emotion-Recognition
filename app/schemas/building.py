from app import db
from app.models.building import BuildingModel
from marshmallow_sqlalchemy import ModelSchema


class BuildingSchema(ModelSchema):
    class Meta:
        model = BuildingModel
