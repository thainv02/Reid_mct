from pymongo import MongoClient
from c.cConst import Const

# Initialize constants and logging
var = Const()

def connect_to_mongo(log_obj):
    """Establishes connection to MongoDB database."""
    try:
        client = MongoClient(var.connection_string)
        db = client[var.client]
        collection = db[var.db]
        if collection.estimated_document_count() >= 0:
            print("Kết nối thành công tới MongoDB!")
            log_obj.info("Kết nối thành công tới MongoDB!")
        return collection
    except Exception as e:
        print("Kết nối thất bại:", e)
        log_obj.info("Kết nối DB thất bại:" + " " + str(e))
        return None