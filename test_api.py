#!/usr/bin/env python3
"""
Simple test script to verify the API endpoints work correctly
"""
import requests
import json
import time

def test_health_endpoint(base_url="http://localhost:7860"):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/api/health", timeout=30)
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

def test_generate_endpoint(base_url="http://localhost:7860"):
    """Test the generate endpoint"""
    try:
        payload = {
            "prompt": "a simple ceramic pottery vase",
            "steps": 10,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        }
        
        response = requests.post(
            f"{base_url}/api/generate", 
            json=payload, 
            timeout=120
        )
        
        print(f"Generate endpoint status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Generated image for prompt: {result.get('prompt')}")
            print(f"Parameters used: {result.get('parameters')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Generate test failed: {e}")
        return False

def test_root_endpoint(base_url="http://localhost:7860"):
    """Test the root endpoint"""
    try:
        response = requests.get(f"{base_url}/", timeout=30)
        print(f"Root endpoint status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Root endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing API endpoints...")
    
    # Test root endpoint
    print("\n1️⃣ Testing root endpoint...")
    test_root_endpoint()
    
    # Test health endpoint
    print("\n2️⃣ Testing health endpoint...")
    test_health_endpoint()
    
    # Wait a bit for model to load
    print("\n⏳ Waiting for model to load...")
    time.sleep(5)
    
    # Test generate endpoint
    print("\n3️⃣ Testing generate endpoint...")
    test_generate_endpoint()
    
    print("\n✅ Tests completed!")
