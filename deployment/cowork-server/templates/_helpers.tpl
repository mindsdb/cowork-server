{{/*
The ports a NetworkPolicy rule admits: the port the API container listens on
and every extra container port (the draft-preview listener). Both come from the
values the deployment itself renders, so the two cannot drift: a policy naming
a port nothing serves drops every legitimate request while `helm template`
stays clean. Every rule admits the same ports.
*/}}
{{- define "cowork-server.policyPorts" -}}
- protocol: TCP
  port: {{ .Values.networkPolicy.port | default .Values.deployment.service.targetPort }}
{{- range .Values.deployment.additionalPorts }}
- protocol: {{ .protocol | default "TCP" }}
  port: {{ .containerPort }}
{{- end }}
{{- end }}
