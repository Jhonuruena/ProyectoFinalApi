using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;
using Microsoft.OpenApi.Models;
using Microsoft.ML.Data;
using System.Net.Http;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", builder =>
    {
        builder.AllowAnyOrigin()
               .AllowAnyMethod()
               .AllowAnyHeader();
    });
});


builder.Services.AddPredictionEnginePool<ClasificacionLetras.ModelInput, ClasificacionLetras.ModelOutput>()
    .FromFile("ClasificacionLetras.mlnet");
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "Sign Language Translator API", Description = "API para traducir lenguaje de señas a texto", Version = "v1" });
});

var app = builder.Build();
app.UseCors("AllowAll");


app.UseSwagger();
if (app.Environment.IsDevelopment())
{
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "Sign Language Translator API V1");
    });
}


app.MapPost("/predict-sequence", async (PredictionEnginePool<ClasificacionLetras.ModelInput, ClasificacionLetras.ModelOutput> predictionEnginePool, List<string> imageUrls) =>
{
    List<string> predictedLetters = new List<string>();

    using (HttpClient client = new HttpClient())
    {
        foreach (var imageUrl in imageUrls)
        {
            try
            {
                var imageBytes = await client.GetByteArrayAsync(imageUrl);

                ClasificacionLetras.ModelInput sampleData = new ClasificacionLetras.ModelInput()
                {
                    ImageSource = imageBytes,
                };

                var sortedScoresWithLabel = ClasificacionLetras.PredictAllLabels(sampleData);

                var highestScore = sortedScoresWithLabel.First();

                predictedLetters.Add(highestScore.Key);
            }
            catch (Exception ex)
            {
                predictedLetters.Add($"Error: {ex.Message}");
            }
        }
    }

    string predictedWord = string.Join("", predictedLetters);

    return await Task.FromResult(predictedWord);
});

app.Run();
