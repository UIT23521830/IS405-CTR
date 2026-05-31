from __future__ import annotations

AVAZU_COLUMNS = [
    "id",
    "click",
    "hour",
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]

TARGET_COL = "click"
ID_COL = "id"

DTYPE_MAP = {
    "click": "int8",
    "C1": "int16",
    "banner_pos": "int8",
    "device_type": "int8",
    "device_conn_type": "int8",
    "C14": "int32",
    "C15": "int16",
    "C16": "int16",
    "C17": "int16",
    "C18": "int8",
    "C19": "int16",
    "C20": "int32",
    "C21": "int16",
}

