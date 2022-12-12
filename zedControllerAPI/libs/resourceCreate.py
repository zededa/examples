import json
import uuid


def constructDs(args):
    print("=" * 100)
    zmethod = "Datastore Create global object"
    print(zmethod.center(70))
    print("=" * 100)
    secrets = {
        "apiKey": None,
        "apiPasswd": None
    }
    payload = {}
    payload['name'] = args["<name>"]
    payload['dsType'] = "DATASTORE_TYPE_" + args["<type>"].upper()
    payload['title'] = payload["name"]
    payload['region'] = None
    secret = {}

    if args["<type>"].upper() == "AWSS3":
        if "--region" not in args:
            print("ERROR: Specify AWS S3 region")
            return 1, payload
        payload['dsFQDN'] = f"https://s3-{args['--region']}.amazon.com"
    elif '--fqdn' in args:
        payload['dsFQDN'] = args['--fqdn']

    if '--dpath' in args:
        payload['dsPath'] = args['--dpath']

    if '--apikey' in args:
        secrets['apiKey'] = args['--apikey']

    if '--apipass' in args:
        secrets['apiPasswd'] = args['--apipass']

    payload['secret'] = secrets
    if '--origin-type' in args:
        payload['originType'] = "ORIGIN_" + args['--origin-type'].upper()
    if '--description' in args:
        payload['description'] = args['--description']

    return 0, payload


def imageCreate(args, dsId):
    print("=" * 100)
    zmethod = "Image Create global object"
    print(zmethod.center(70))
    print("=" * 100)

    payload = {}
    payload['name'] = args['<name>']
    payload['datastoreId'] = dsId
    payload['imageFormat'] = 'QCOW2'
    payload['imageType'] = 'IMAGE_TYPE_APPLICATION'
    payload['imageArch'] = "AMD64"
    payload['title'] = args['<name>']

    if '--origin-type' in args:
        payload['originType'] = "ORIGIN_" + args['--origin-type'].upper()

    if '--description' in args:
        payload['description'] = args['--description']

    return 0, payload


def imageUplink(zsession, args):
    print("=" * 100)
    zmethod = "Image Uplink"
    print(zmethod.center(70))
    print("=" * 100)

    payload = {}
    getUrl = f"/api/v1/apps/images/name/{args['<name>']}"
    status, response = zsession.get_request(getUrl)
    if status != 0:
        print(f"get image {args['<name>']} failed")
        return 1

    putUrl = f"/api/v1/apps/images/id/{response['id']}/uplink"
    payload = response
    payload['imageSha256'] = args['--image-sha']
    payload['imageSizeBytes'] = args['--image-size']
    status, response = zsession.put_request(putUrl, payload)
    if status != 0:
        print(f"PUT request to uplink image {args['<name>']} failed")
        return 1

    return 0


def edgeAppCreate(zsession, args):
    print("=" * 100)
    zmethod = "Edge-app create Global Object"
    print(zmethod.center(70))
    print("=" * 100)

    createUrl = "/api/v1/apps"
    payload = {}
    payload['name'] = args['<name>']
    payload['title'] = args['<name>']
    with open(args['--manifest'], 'r') as f:
        data = json.load(f)
    payload['manifestJSON'] = data
    if '--origin-type' in args:
        payload['originType'] = "ORIGIN_" + args['--origin-type'].upper()

    if '--description' in args:
        payload['description'] = args['--description']

    status, response = zsession.post_request(createUrl, payload)
    if status != 0:
        print(f"create edge-app {args['<name>']} failed response {response}")
        return 1
    return 0


def updateDataStore(zsession, args):
    print("=" * 100)
    zmethod = "DataStore update to Global Object"
    print(zmethod.center(70))
    print("=" * 100)

    getUrl = f"/api/v1/datastores/name/{args['<name>']}"
    status, response = zsession.get_request(getUrl)
    if status != 0:
        print(f"Get datastore resource failed datastore name {args['<name>']}")
        return 1
    # Convert originType to global
    payload = response
    payload['originType'] = "ORIGIN_GLOBAL"
    putUrl = f"/api/v1/datastores/id/{response['id']}"
    zsession.put_request(putUrl, payload)
    if status != 0:
        print(f"PUT request to update originType failed {args['<name>']}")
        return 1
    return 0


def updateImage(zsession, args):
    print("=" * 100)
    zmethod = "Image update to Global Object"
    print(zmethod.center(70))
    print("=" * 100)

    getUrl = f"/api/v1/apps/images/name/{args['<name>']}"
    status, response = zsession.get_request(getUrl)
    if status != 0:
        print(f"Get image failed {args['<name>']} response {response}")
        return 1
    payload = response
    payload['originType'] = "ORIGIN_GLOBAL"
    putUrl = f"/api/v1/apps/images/id/{response['id']}"
    status, response = zsession.put_request(putUrl, payload)
    if status != 0:
        print(f"Update image {args['<name>']} failed response {response}")
        return 1
    return 0


def updateEdgeApp(zsession, args):
    print("=" * 100)
    zmethod = "Edge-update update to Global Object"
    print(zmethod.center(70))
    print("=" * 100)

    getUrl = f"/api/v1/apps/name/{args['<name>']}"
    status, response = zsession.get_request(getUrl)
    if status != 0:
        print(f"Get edge-app failed {args['<name>']} response {response}")
        return 1

    payload = response
    payload['originType'] = "ORIGIN_GLOBAL"
    putUrl = f"/api/v1/apps/id/{response['id']}"
    status, response = zsession.put_request(putUrl, payload)

    if status != 0:
        print(f"Update edge-app {args['<name>']} failed response {response}")
        return 1
    return 0


def jobEdgeApp(zsession, appName, serviceType):
    data = {}
    payload = {}
    jobUUID = str(uuid.uuid4())
    data['name'] = f"update-{serviceType}-{appName}-{jobUUID}"
    api = f"/api/v1/jobs"
    if serviceType == 'import':
        data['operationType'] = "BULK_SERVICE_BUNDLE_IMPORT"
    data["objectType"] = "OBJECT_TYPE_EDGE_APP"
    status, response = zsession.post_request(api, data)
    if status != 0:
        print(f"Create {serviceType} job failed responses {response}")
        return 1, None

    return 0, data['name']


def edgeAppRefresh(zsession, args):
    print("=" * 100)
    zmethod = "Edge-app update refresh to global object"
    print(zmethod.center(70))
    print("=" * 100)

    getUrl = f"/api/v1/apps/name/{args['<name>']}"
    importUrl = f"/api/v1/jobs/apps/bundles/import"

    importPayload = {
        "bundleImport": {
            "bundleConfig": [{
                "name": args['<name>'],
                "parentBundleId": ""
            }]
        },
        "jobName": ""
    }
    status, response = zsession.get_request(getUrl)
    if status != 0:
        print(f"Get edge-app failed {args['<name>']} response {response}")
        return 1

    parentID = response['parentDetail']['idOfParentObject']
    if response['parentDetail']['updateAvailable'] is True:
        jStaus, jName = jobEdgeApp(zsession, args['<name>'], "import")
        if jStaus != 0:
            return 1
        importPayload["bundleImport"]["bundleConfig"][0]["parentBundleId"] = parentID
        importPayload["jobName"] = jName
        impStatus, impResponse = zsession.put_request(importUrl, importPayload)

    return 0
