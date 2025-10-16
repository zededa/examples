{{/*
Expand the name of the chart.
*/}}
{{- define "edge-ai-car-classification.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "edge-ai-car-classification.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "edge-ai-car-classification.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "edge-ai-car-classification.labels" -}}
helm.sh/chart: {{ include "edge-ai-car-classification.chart" . }}
{{ include "edge-ai-car-classification.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "edge-ai-car-classification.selectorLabels" -}}
app.kubernetes.io/name: {{ include "edge-ai-car-classification.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "edge-ai-car-classification.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "edge-ai-car-classification.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Server deployment labels
*/}}
{{- define "edge-ai-car-classification.server.labels" -}}
{{ include "edge-ai-car-classification.labels" . }}
app.kubernetes.io/component: server
{{- end }}

{{/*
Server selector labels
*/}}
{{- define "edge-ai-car-classification.server.selectorLabels" -}}
{{ include "edge-ai-car-classification.selectorLabels" . }}
app.kubernetes.io/component: server
{{- end }}

{{/*
Client deployment labels
*/}}
{{- define "edge-ai-car-classification.client.labels" -}}
{{ include "edge-ai-car-classification.labels" . }}
app.kubernetes.io/component: client
{{- end }}

{{/*
Client selector labels
*/}}
{{- define "edge-ai-car-classification.client.selectorLabels" -}}
{{ include "edge-ai-car-classification.selectorLabels" . }}
app.kubernetes.io/component: client
{{- end }}

{{/*
Create MinIO service name (for external MinIO, this is not used)
*/}}
{{- define "edge-ai-car-classification.minio.serviceName" -}}
{{- if .Values.minio.enabled }}
{{- printf "%s-minio" .Release.Name }}
{{- else }}
{{- printf "external-minio" }}
{{- end }}
{{- end }}

{{/*
Create MinIO endpoint URL
*/}}
{{- define "edge-ai-car-classification.minio.endpoint" -}}
{{- if .Values.minio.enabled }}
{{- printf "http://%s:%d" (include "edge-ai-car-classification.minio.serviceName" .) (.Values.minio.service.port | int) }}
{{- else }}
{{- printf "http://%s" .Values.minio.external.endpoint }}
{{- end }}
{{- end }}

{{/*
Create OVMS service name
*/}}
{{- define "edge-ai-car-classification.ovms.serviceName" -}}
{{- printf "%s-ovms" (include "edge-ai-car-classification.fullname" .) }}
{{- end }}

{{/*
Create OVMS REST endpoint URL
*/}}
{{- define "edge-ai-car-classification.ovms.restEndpoint" -}}
{{- printf "http://%s:%d" (include "edge-ai-car-classification.ovms.serviceName" .) (.Values.service.ovms.ports.rest | int) }}
{{- end }}

{{/*
Image name with registry
*/}}
{{- define "edge-ai-car-classification.image" -}}
{{- if .registry }}
{{- printf "%s/%s" .registry .image }}
{{- else }}
{{- .image }}
{{- end }}
{{- end }}
