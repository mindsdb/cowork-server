# Cowork Server API — Design Notes

## Architectural Overview

Given below is a high-level architectural overview of the Cowork Server API, specifically outlining how it differs from the implementation already available in the [mindsdb/cowork](https://github.com/mindsdb/cowork) repository. Some of these decisions have been made in order to simplify the onboarding of other agents (harnesses) such as Hermes.

Here is a breakdown:
### App Vs Harness Components
At the moment, most components of the existing Cowork server including projects, conversations, attachments etc., are closely coupled with the Anton agent. This means that when other agents (e.g., Hermes) are onboarded, they will have to either adhere to the structure Anton has defined for these components or implement their own versions.

A good example is the the work that has been done [here](https://github.com/mindsdb/cowork/compare/main...hermes-mvp); the base abstraction for harnesses here has been defined in such a way that conversation management and other aspects need to be implemented separately for each agent. This is not ideal as it leads to code duplication and makes maintenance harder.

Furthermore, it is not entirely clear how harness-specific components such as memory and skills work here. It seems to be necessary to use the contract defined by Anton and it is not guaranteed that these will work as intended for other agents.

A better way to approach this would be to have a clear separation between the app and harness components. The app components (e.g., projects, conversations, attachments) should be designed in a way that they are independent of any specific agent. We were already able to achieve in our implementation of the Minds API, where different implementations of agents could be onboarded with minimal changes to the core API.

The implementation available here aims to achieve this by taking parts of the both the existing codebase and the Minds API.

### Database Design
This implementation has also been designed to allow for database storage rather than relying on a file-based storage system. A lightweight SQLite database can be used here, but it is also able to support more robust databases such as Postgres if needed. This allows for better scalability and performance, especially as the number of users and conversations grows.

### API Design
The design of the API has also been hardened by removing several unusued endpoints and improving on the existing ones.

For example, the Responses API has been updated to allow for file inputs along with support for an OpenAI-compatible Files API. This alleviates the need for maintaining the /attachments endpoints defined in the orignial server defines a standard way for handling file uploads and attachments across different agents. More information regarding these design updates can be found in the design document linked below.

Further details regarding this design can be found in this document: [Cowork Server API for Agents](https://docs.google.com/document/d/1YBgr59GoO47wvLtZAO7wbNL8DKrigww_PYeUlcMDgos/edit?usp=sharing).
