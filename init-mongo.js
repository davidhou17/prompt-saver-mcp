// MongoDB initialization script
// This script creates the database and sets up initial indexes

db = db.getSiblingDB('prompt_saver');

// Create the prompts collection
db.createCollection('prompts');

// Create indexes for better performance
db.prompts.createIndex({ "use_case": 1 });
db.prompts.createIndex({ "last_updated": -1 });
db.prompts.createIndex({ 
  "summary": "text", 
  "prompt_template": "text" 
}, { 
  name: "text_search_index" 
});

// For vector search (if using MongoDB Atlas)
// This would be created through the Atlas UI or Atlas CLI
// db.prompts.createSearchIndex({
//   "name": "vector_index",
//   "definition": {
//     "fields": [
//       {
//         "type": "vector",
//         "path": "embedding",
//         "numDimensions": 1536,  // for text-embedding-3-small
//         "similarity": "cosine"
//       }
//     ]
//   }
// });

print('Database initialized successfully');
