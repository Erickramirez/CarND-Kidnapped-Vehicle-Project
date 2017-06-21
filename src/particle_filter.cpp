/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
# define M_PI           3.14159265358979323846  /* pi */

using namespace std;

std::normal_distribution<double> g_distribution(0,1);
std::default_random_engine gen;
std::mt19937 generator(gen());

double Multivariate_Gaussian(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
    return exp(-((x-mu_x)*(x-mu_x)/(2*sig_x*sig_x) + (y-mu_y)*(y-mu_y)/(2*sig_y*sig_y))) / (2.0*M_PI*sig_x*sig_y);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles =200;
	weights.resize(num_particles);
	for (int i = 0; i<num_particles;i++){
        Particle p;
	    p.id = i;
        p.x=x+std[0] *g_distribution(generator);
        p.y=y+std[1] *g_distribution(generator);
        p.theta=x+std[2] *g_distribution(generator);
        p.weight = 1.0f;
        weights[i] = 1.0f;
        particles.push_back(p);

    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    for (int i = 0; i < num_particles; i++) {
        double th = particles[i].theta;
        double v = velocity;
        double dth = yaw_rate;

       if (fabs(dth)>0.001){
            particles[i].x +=   v/dth*(sin(th+dth*delta_t) - sin(th))+ std_pos[0]*g_distribution(generator);
            particles[i].y +=  -v/dth*(cos(th+dth*delta_t) - cos(th))+ std_pos[1]*g_distribution(generator);
            particles[i].theta = fmod(particles[i].theta + dth*delta_t+ std_pos[2]*g_distribution(generator),2.0*M_PI);
        } else {
            particles[i].x += v*cos(th)*delta_t+ std_pos[0]*g_distribution(generator);
            particles[i].y += v*sin(th)*delta_t+ std_pos[1]*g_distribution(generator);
        }

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    double distance;
    for (int i=0;i<observations.size();i++){ // loop over observations
        double min_dist = 1000000;
        int closest_id = -1;
        for (int j=0;j<predicted.size();j++){
            distance = dist(observations[i].x, observations[i].y,
                           predicted[j].x, predicted[j].y);
            if (distance<min_dist){
                min_dist = distance;
                closest_id = predicted[j].id;
            }
        }
        observations[i].id  = closest_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    weights.clear();
    for (int i=0;i<particles.size();i++) {
        // Convert observations to ground frame
        std::vector<LandmarkObs> observations_grnd;
        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs obsv_i;
            obsv_i.x = (observations[j].x * cos(particles[i].theta)) + (-observations[j].y * sin(particles[i].theta)) +
                        particles[i].x;
            obsv_i.y = (observations[j].x * sin(particles[i].theta)) + (observations[j].y * cos(particles[i].theta)) +
                        particles[i].y;
            obsv_i.id = -1; // Id not assigned
            observations_grnd.push_back(obsv_i);
        }


        std::vector<LandmarkObs> predicted_meas;
        // Compute predicted measurements
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            double distance_particle_obs;
            distance_particle_obs = dist(particles[i].x, particles[i].y,
                                         map_landmarks.landmark_list[j].x_f,
                                         map_landmarks.landmark_list[j].y_f);
            if (distance_particle_obs <= sensor_range) {
                LandmarkObs predicted_i;
                predicted_i.id = map_landmarks.landmark_list[j].id_i;
                predicted_i.x = map_landmarks.landmark_list[j].x_f;
                predicted_i.y = map_landmarks.landmark_list[j].y_f;
                predicted_meas.push_back(predicted_i);
            }
        }

        dataAssociation(predicted_meas, observations_grnd);
        double prob = 1.0;
        double prob_i;
        for (int j = 0; j < predicted_meas.size(); j++) {
            int ind_min = -1;
            double dist_min = 1000000;
            for (int k = 0; k < observations_grnd.size(); k++) {
                if (predicted_meas[j].id == observations_grnd[k].id) {
                    double check_dist = dist(predicted_meas[j].x,
                                             predicted_meas[j].y,
                                             observations_grnd[k].x,
                                             observations_grnd[k].y);
                    if (check_dist < dist_min) {
                        ind_min = k;
                        dist_min = check_dist;
                    }
                }
            }
            if (ind_min != -1) {
                prob_i =Multivariate_Gaussian(predicted_meas[j].x, predicted_meas[j].y,
                                          observations_grnd[ind_min].x, observations_grnd[ind_min].y,
                                          std_landmark[0], std_landmark[1]);

                prob = prob * prob_i;
            }


        }
        weights.push_back(prob);
        particles[i].weight = prob;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::random_device random_weights;
    std::mt19937 generator_weights(random_weights());

    // Creates a discrete distribution for weight.
    std::discrete_distribution<int> distribution_weights(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles;
    // Resampling
    for(int i=0;i<num_particles;i++){
        Particle particles_i = particles[distribution_weights(generator_weights)];
        resampled_particles.push_back(particles_i);
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
