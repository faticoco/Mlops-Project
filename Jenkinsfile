pipeline {
    agent any
    
    environment {
        BACKEND_APP_NAME = 'house-pred-backend'
        FRONTEND_APP_NAME = 'house-pred-frontend'
        DOCKER_USERNAME = 'ahmed93560'
        BACKEND_DOCKER_IMAGE = "${DOCKER_USERNAME}/${BACKEND_APP_NAME}"
        FRONTEND_DOCKER_IMAGE = "${DOCKER_USERNAME}/${FRONTEND_APP_NAME}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Backend Docker Image') {
            steps {
                dir('src') {
                    bat "docker build -t ${BACKEND_DOCKER_IMAGE} ."
                }
            }
        }
        
        stage('Tag Backend Docker Image') {
            steps {
                bat "docker tag ${BACKEND_DOCKER_IMAGE} ${DOCKER_USERNAME}/${BACKEND_APP_NAME}:latest"
            }
        }
        
        stage('Build Frontend Docker Image') {
            steps {
                dir('frontend') {
                    bat "docker build -t ${FRONTEND_DOCKER_IMAGE} ."
                }
            }
        }
        
        stage('Tag Frontend Docker Image') {
            steps {
                bat "docker tag ${FRONTEND_DOCKER_IMAGE} ${DOCKER_USERNAME}/${FRONTEND_APP_NAME}:latest"
            }
        }
        
        stage('Push Docker Images') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-cred', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    bat 'docker login -u %DOCKER_USER% -p %DOCKER_PASS%'
                    bat "docker push ${DOCKER_USERNAME}/${BACKEND_APP_NAME}:latest"
                    bat "docker push ${DOCKER_USERNAME}/${FRONTEND_APP_NAME}:latest"
                }
            }
        }
    }
}