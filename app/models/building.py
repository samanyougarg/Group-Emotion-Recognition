from .. import db

class BuildingModel(db.Model):

    __tablename__ = 'BUILDING'

    BUILDINGID = db.Column(db.Integer, primary_key=True)
    BUILDINGNAME = db.Column(db.String(255), nullable=False)
    BUILDINGCITY = db.Column(db.String(255))
    BUILDINGSTATE = db.Column(db.String(255))
    BUILDINGCOUNTRY = db.Column(db.String(255))


    @classmethod
    def find_by_building_id(cls, BUILDINGID):
        return cls.query.filter_by(BUILDINGID=BUILDINGID).first()
