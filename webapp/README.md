# REST API

This project aims to provide an example of an API that follows some of the Best Practices for designing a REST API.

## REST API Best Practices

This page lists out some of the Best Practices and Guidelines that should be followed when designing a REST API to make it easy for the developers to use and implement.

### 1\. Versioning the API

Upgrading your API with a breaking change may break the existing products/services using your API. Versioning ensures that your users can still access the older version of your API for some time while they transition to the newer version.  
Example - `http://example.com/api/v1/buildings`

### 2\. HTTP Methods

HTTP methods define the types of actions that can be performed on a resource.

*   **GET** method retrieves data from a resource.  
    Examples -
    *   `GET /buildings` will get a list of all the buildings.
        *   `GET /buildings/1` will get the building with ID 1.
*   **POST** method requests the server to insert a resource into the database.  
    Examples -
    *   `POST /buildings` can be used to insert one or more buildings into the database.
*   **PUT** method requests the server to update a resource or create it if it doesn't exist.  
    Examples -
    *   `PUT /buildings/1` will update the building with ID 1 or create it if it doesn't exist.
*   **PATCH** method requests the server to partially update a resource in the database.  
    Examples -
    *   `PATCH /buildings/1` will update the building with ID 1.
*   **DELETE** method requests the server to delete a resource from the database.  
    Examples -
    *   `DELETE /buildings/1` can be used to delete the building with ID 1.

### 3\. RESTful Endpoints

*   Endpoints should include nouns, not verbs or actions.  
    **Avoid**

    *   /getAllBuildings
    *   /getBuilding
    *   /addBuilding
    *   /deleteBuilding

    **Use**

    *   /buildings
*   Only Plural nouns should be used for consistency.

*   But how do you tell the endpoint what action to perform? Solution - Use HTTP verbs (GET, POST, PUT, PATCH, DELETE) to perform CRUD actions.  
    Examples -

    *   `GET /buildings` will get a list of all the buildings.

    *   `GET /buildings/1` will get the building with ID 1.

    *   `POST /buildings` can be used to insert one or more buildings into the database. Do not use `/building` to post a single building. `/buildings` should be able to handle one or more buildings.

    *   `DELETE /buildings/1` will delete the building with ID 1.

    *   `PUT/PATCH /buildings/1` will update the building with ID 1.

### 4\. HTTP Responses

HTTP defines a set of Status Codes that must be included in your API responses to tell the developer that status of his/her request. Here are some of the commonly used Status Codes -

*   `200 OK` - GET, PUT, PATCH or DELETE was successful.
*   `201 Created` - POST request was successful.
*   `204 No Content` - DELETE was successful.
*   `400 Bad Request` - There were inavlid parameters in the request. Developer must fix them.
*   `401 Unauthorized` - Invalid Credentials were provided.
*   `404 Not Found` - URL does not exist.
*   `405 Method Not Allowed` - Requested method is not allowed on the resource.
*   `500 Internal Server Error` - Problem with the server.

### 5\. Documentation

Docs should be easily accessible and readable. Should provide examples of requests and responses. Should always be upto date. Your docs should follow the Open API standards. We have used two types of docs in this example, both of which depend on the Open API spec.

1.  **Swagger UI** - Lets users test out the various endpoints.
2.  **ReDoc** - Easily to read detailed documentation.

### 6\. Validation

Your API should validate the requests being sent to the server and provide appropriate responses. POST, PUT and PATCH requests should provide field specific errors to help the developer locate and fix the issue.

### 7\. SSL

Your API should always use SSL for all endpoints to ensure the safety of the data.
