from datetime import datetime, timedelta


class Response:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.json_data = json_data
        self.elapsed = timedelta(seconds=2)
        self.text = "file content"
        self.headers = "header"
        self.url = "url"
        self.reason = "reason"

    def json(self):
        return self.json_data

    def iter_content(self, **kwargs):
        return [b"file content"]

    def raise_for_status(self):
        if self.status_code > 250:
            raise ValueError("Failed to download file")


def get(url, headers, **kwargs):
    api_key = headers.get("apikey")
    if api_key is None:
        response = Response(403, {"message": "Forbidden"})
        return response
    if str(api_key) not in ["123456", "654321"]:
        response = Response(403, {"message": "Forbidden"})
        return response

    date_now = datetime.now()

    base_url = "https://data.hub.api.metoffice.gov.uk/atmospheric-models/1.0.0"

    if url == base_url + "/runs/mo-uk-latlon?sort=RUNDATETIME":
        json_data = {
            "modelId": "mo-uk-latlon",
            "completeRuns": [
                {
                    "run": "00",
                    "runDateTime": date_now.strftime("%Y-%m-%dT00:00:00Z"),
                    "runFilter": date_now.strftime("%Y%m%d00"),
                },
            ],
        }
        return Response(200, json_data)

    if base_url + "/orders" in url:
        if url == base_url + "/orders?detail=MINIMAL":
            if api_key == "654321":
                json_data = {"orders": []}
            else:
                json_data = {
                    "orders": [
                        {
                            "orderId": "123456",
                            "name": "123456",
                            "modelId": "mo-uk-latlon",
                            "requiredLatestRuns": [
                                "00",
                                "01",
                                "02",
                                "03",
                                "04",
                                "05",
                                "06",
                                "08",
                                "09",
                                "10",
                                "11",
                                "12",
                                "13",
                                "14",
                                "15",
                                "16",
                                "17",
                                "18",
                                "19",
                                "20",
                                "21",
                                "22",
                                "23",
                            ],
                            "format": "GRIB2",
                        }
                    ]
                }
            response = Response(200, json_data)
            return response
        if url == base_url + "/orders/123456/latest?detail=MINIMAL&runfilter=00":
            json_data = {
                "orderDetails": {
                    "order": {
                        "orderId": "123456",
                        "name": "123456",
                        "modelId": "mo-uk-latlon",
                        "format": "GRIB2",
                    },
                    "files": [
                        {
                            "fileId": "agl_u-component-of-wind-surface-adjusted_10.0_+00",
                            "runDateTime": date_now.strftime("%Y-%m-%dT00:00:00Z"),
                            "run": "0",
                        },
                        {
                            "fileId": "agl_v-component-of-wind-surface-adjusted_10.0_+00",
                            "runDateTime": date_now.strftime("%Y-%m-%dT00:00:00Z"),
                            "run": "0",
                        },
                    ],
                }
            }
            response = Response(200, json_data)
            return response
        if base_url + "/orders/123456/latest/" in url:
            filenames = [
                "agl_u-component-of-wind-surface-adjusted_10.0_%2B00",
                "agl_v-component-of-wind-surface-adjusted_10.0_%2B00",
            ]
            for filename in filenames:
                if url == base_url + f"/orders/123456/latest/{filename}/data":
                    json_data = {
                        "modelId": "mo-uk-latlon",
                        "run": "00",
                        "runDateTime": date_now.strftime("%Y-%m-%dT00:00:00Z"),
                        "runFilter": date_now.strftime("%Y%m%d00"),
                    }
                    return Response(200, json_data)

    response = Response(404, {"message": "Not Found"})
    return response
